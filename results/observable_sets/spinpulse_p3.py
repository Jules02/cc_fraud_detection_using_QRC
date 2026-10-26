"""P3: memory capacity of the observable sets on the pulse-level SpinPulse model, with the basis changes as pulses.

Same task, reservoir, observable sets, shot model, readout and output format as observables.py (P2); only the noise
model changes. The reservoir and the RY(pi u) encoding are compiled to SpinPulse pulses (SpinPulseBackend's
timeline). Each noise realization is one set of pink (1/f) dephasing traces covering the whole window plus a
measurement block, shared by all windows. With the traces fixed every block is a unitary, and the state is
propagated as a density matrix, so the input reset is exact (no sampled reset trajectories). The measurement block
applies each setting's frame change as pulses on the same traces: per qubit RZ(lambda) then RY(theta) of the ZYZ
decomposition of the frame unitary (the last RZ commutes with the Z readout); qubits measured in Z idle through the
block. Outcome distributions are averaged over the realizations; readout flips (0.07 %) and the reset preparation
error are applied as in P2.

Calibration (--calibrate). SpinPulse noise is pure dephasing, whose infidelity grows as (duration / T2*)^2, and a
two-qubit RZZ (with its echo) is ~20x longer than a 1q pi/2 pulse. So one T2* cannot give the hardware's 98 % 2q
and 99.6 % 1q at once: T2* is set so that RZZ(pi/2) has the target average gate fidelity, and the 1q fidelity
that follows (~99.998 %) is reported. The missing 1q error (control errors) is not in SpinPulse.

Usage:
    python results/observable_sets/spinpulse_p3.py --calibrate
    python results/observable_sets/spinpulse_p3.py --check
    python results/observable_sets/spinpulse_p3.py --n 5 --seed 42 --out results/observable_sets/data/p3
"""
import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, PauliList, random_unitary
from qiskit.synthesis import OneQubitEulerDecomposer

import observables as obs
from qrc.backends.numpy_backend import NumpyBackend
from qrc.backends.spinpulse import (
    SpinPulseBackend,
    SpinPulseConfig,
    _circuit_unitary,
    _idle_unitary,
    _propagate_rotations,
    _trace_slice,
    _unit_pulse,
)
from qrc.encodings import AngleEncoding
from qrc.protocol import QRCProtocol
from spin_pulse import PulseCircuit

P_RO = 0.0007
REALIZATIONS = 50
# name -> pink-noise T2* (None = no noise), from --calibrate: RZZ(pi/2) average gate fidelity 99 % / 98 %
LEVELS = {
    "sp-ideal": None,
    "sp-2q99": 156.0,
    "sp-2q98": 110.0,
}
LEVEL_ID_OFFSET = 100  # sampling-stream ids distinct from P2's levels
_ZYZ = OneQubitEulerDecomposer("ZYZ")


# ---------------------------------------------------------------- calibration

def gate_fidelity(qc, config, realizations=400):
    """Average gate fidelity of the channel SpinPulse gives `qc` (mean over noise realizations)."""
    backend = SpinPulseBackend(config)
    n = qc.num_qubits
    specs = backend._hardware_specs(n)
    pulses = PulseCircuit.from_circuit(specs.gate_transpile(qc), hardware_specs=specs, exp_env=None)
    ideal = Operator(qc).data
    env = backend._environment(specs, pulses.duration)
    d = 2 ** n
    F = []
    for r in range(realizations if env is not None else 1):
        if env is not None:
            if r > 0:
                env.generate_time_traces()
            pulses.t_lab = 0
            pulses.attach_time_traces(env)
        U = _circuit_unitary(pulses.to_circuit())
        F.append(abs(np.trace(ideal.conj().T @ U)) ** 2 / d ** 2)
    return (d * np.mean(F) + 1) / (d + 1)


def calibrate():
    q1 = QuantumCircuit(1)
    q1.rx(np.pi / 2, 0)
    q2 = QuantumCircuit(2)
    q2.rzz(np.pi / 2, 0, 1)
    for target in (0.99, 0.98):
        lo, hi = 50.0, 1000.0  # 2q fidelity grows with T2*; bisect in log T2*
        for _ in range(12):
            mid = np.sqrt(lo * hi)
            if gate_fidelity(q2, SpinPulseConfig(noise_type="pink", T2S=mid, seed=0)) < target:
                lo = mid
            else:
                hi = mid
        T2S = np.sqrt(lo * hi)
        f2 = gate_fidelity(q2, SpinPulseConfig(noise_type="pink", T2S=T2S, seed=1))
        f1 = gate_fidelity(q1, SpinPulseConfig(noise_type="pink", T2S=T2S, seed=1))
        print(f"target 2q {target}: T2* = {T2S:.1f}  ->  RZZ(pi/2) {f2:.4f}, RX(pi/2) {f1:.5f} (fresh seed)")


# ---------------------------------------------------------------- measurement frames

def qubit_factors(key, n, seed):
    """Per-qubit 2x2 frame unitaries of a setting, index = qubit."""
    if key.startswith("frame"):
        rng = np.random.default_rng(10_000 + seed)  # same draws as obs.pauli_settings
        draws = [[random_unitary(2, seed=rng).data for _ in range(n)] for _ in range(2)]
        return draws[int(key[5:])][::-1]  # kron order there is qubit n-1 first
    if key == "Z":
        return [np.eye(2)] * n
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    Sdg = np.diag([1, -1j])
    single = {"X": H, "Y": H @ Sdg, "Z": np.eye(2)}
    return [single[b] for b in reversed(key)]  # basis strings are little-endian


def frame_gates(factors):
    """Per qubit, the measurement-block gate list for _propagate_rotations: RZ(lam) in slot 0, RY(theta) in slot 1."""
    out = []
    for U in factors:
        theta, _, lam = _ZYZ.angles(U)
        out.append([(0, "z", None, lam), (1, "y", None, theta)])
    return out


# ---------------------------------------------------------------- simulation

def simulate(X, reservoir, n, T2S, seed, keys, realizations=REALIZATIONS, meas_block=True, reset=True):
    """Outcome distributions per setting key, averaged over noise realizations, before readout flips.
    `reset=False` is only used by check(), to compare with SpinPulseBackend's exact (trajectory-free) path."""
    noisy = T2S is not None
    config = SpinPulseConfig(noise_type="pink" if noisy else "none", T2S=T2S or 100.0, seed=seed)
    backend = SpinPulseBackend(config)
    tl = backend._timeline_for(AngleEncoding(num_qubits=1), reservoir, 1)
    M, L = X.shape[:2]
    D = 2 ** n
    shapes = {axis: _unit_pulse(axis, tl.specs) for axis in "zy"}
    slots = [(0, len(shapes["z"]), {"z": shapes["z"]}), (len(shapes["z"]), len(shapes["y"]), {"y": shapes["y"]})]
    meas = len(shapes["z"]) + len(shapes["y"]) if meas_block else 0
    gates = {key: frame_gates(qubit_factors(key, n, seed)) for key in keys}
    t_end = L * tl.step_duration
    env = backend._environment(tl.specs, t_end + meas)
    only_idle = env is not None and env.only_idle

    probs = {key: np.zeros((M, D)) for key in keys}
    R = realizations if noisy else 1
    for r in range(R):
        if env is not None and r > 0:
            env.generate_time_traces()
        rho = np.zeros((M, D, D), complex)
        rho[:, 0, 0] = 1
        for t in range(L):
            t_enc = t * tl.step_duration
            if reset and t > 0:
                v = rho.reshape(M, D // 2, 2, D // 2, 2)
                tr = v[:, :, 0, :, 0] + v[:, :, 1, :, 1]
                rho = np.zeros_like(rho)
                v = rho.reshape(M, D // 2, 2, D // 2, 2)
                v[:, :, 0, :, 0] = (1 - P_RO) * tr
                v[:, :, 1, :, 1] = P_RO * tr
            # encoding block: per-window pulse on the input qubit (qubit 0), idle evolution on the memory qubits;
            # the idle part is window independent and folds into the reservoir unitary
            noise = _trace_slice(env, tl.physical[0], t_enc, tl.enc_duration)
            G = _propagate_rotations(tl.gates[0], tl.slots, X[:, t], noise, tl.enc_duration, only_idle)
            idle = np.eye(1)
            for q in range(n - 1, 0, -1):  # kron order: qubit n-1 first
                noise = _trace_slice(env, tl.physical[q], t_enc, tl.enc_duration)
                idle = np.kron(idle, _idle_unitary(noise, tl.enc_duration, tl.specs))
            W = backend._reservoir_unitary(tl, env, t_enc + tl.enc_duration) @ np.kron(idle, np.eye(2))
            step = W @ np.einsum("ab,mij->maibj", np.eye(D // 2), G).reshape(M, D, D)  # W (I (x) G_m)
            rho = step @ rho @ step.conj().transpose(0, 2, 1)
        for key in keys:
            F = np.eye(1)
            for q in range(n - 1, -1, -1):
                if meas:
                    noise = _trace_slice(env, tl.physical[q], t_end, meas)
                    U = _propagate_rotations(gates[key][q], slots, np.zeros((1, 1)), noise, meas, False)[0]
                else:
                    U = np.eye(2)
                F = np.kron(F, U)
            # diag(F rho F^dagger) without forming the full product: sum_a F[x,a] (rho F^dagger)[m,a,x]
            probs[key] += np.einsum("xa,max->mx", F, rho @ F.conj().T).real / R
    return probs


def run(n, seed, length, levels, budgets):
    reservoir = obs.make_reservoir(n, seed)
    sets = obs.pauli_settings(n, seed)
    keys = sorted({key for s in sets.values() for key, *_ in s})
    X_train, Y_train, X_test, Y_test = obs.task_data(seed, length)
    rows, feat_rows = [], []
    for level in levels:
        t0 = time.time()
        probs = {}
        for split, X in (("train", X_train), ("test", X_test)):
            raw = simulate(X, reservoir, n, LEVELS[level], seed, keys)
            probs[split] = {key: obs.readout(p, P_RO, n) for key, p in raw.items()}
        level_id = LEVEL_ID_OFFSET + list(LEVELS).index(level)
        r, f = obs.evaluate(probs, sets, n, seed, level, level_id, budgets, Y_train, Y_test)
        rows += r
        feat_rows += f
        print(f"n={n} seed={seed} {level}: {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(feat_rows)


def check():
    """1. frame factors rebuild the settings' frame unitaries; 2. noiseless pulses: every setting's exact features
    equal SpinPulseBackend's exact Pauli expectations; 3. pink noise, Z readout: the density-matrix loop equals
    SpinPulseBackend on the same traces (2 and 3 without reset, where SpinPulseBackend has no reset trajectories);
    4. noiseless pulses with reset vs the ideal circuit."""
    for n in (4, 5):
        for key, V, *_ in itertools.chain.from_iterable(obs.pauli_settings(n, 42).values()):
            f = qubit_factors(key, n, 42)
            K = np.array([[1.0]])
            for U in f[::-1]:
                K = np.kron(K, U)
            assert np.allclose(K, V), f"frame factors wrong for {key}"
    n = 4
    X = obs.task_data(42, 40)[0][:12]
    reservoir = obs.make_reservoir(n, 42)
    sets = obs.pauli_settings(n, 42)
    keys = sorted({key for s in sets.values() for key, *_ in s})

    probs = simulate(X, reservoir, n, None, 42, keys, reset=False)
    for name in ("XYZ-strings", "XYZ+XYZ-2body", "all-Pauli"):
        settings = sets[name]
        paulis = PauliList([p for *_, group in settings for p in group])
        ref = QRCProtocol(AngleEncoding(1), reservoir, paulis,
                          SpinPulseBackend(SpinPulseConfig(noise_type="none", seed=42))).run(X)
        err = np.abs(obs.features(probs, settings, n, None, None) - ref).max()
        assert err < 1e-6, f"noiseless frame pulses disagree with SpinPulseBackend for {name}: {err:.2e}"

    z = sets["Z-strings"]
    paulis = PauliList([p for *_, group in z for p in group])
    probs = simulate(X, reservoir, n, 105.0, 42, ["Z"], realizations=5, meas_block=False, reset=False)
    ref = QRCProtocol(AngleEncoding(1), reservoir, paulis, SpinPulseBackend(
        SpinPulseConfig(noise_type="pink", T2S=105.0, noise_samples=5, seed=42))).run(X)
    err = np.abs(obs.features(probs, z, n, None, None) - ref).max()
    assert err < 1e-8, f"noisy density-matrix loop disagrees with SpinPulseBackend: {err:.2e}"

    probs = simulate(X, reservoir, n, None, 42, ["Z"])
    ref = QRCProtocol(AngleEncoding(1), reservoir, paulis, NumpyBackend(reset_inputs=True)).run(X)
    err = np.abs(obs.features(probs, z, n, None, None) - ref).max()
    print(f"noiseless pulses + reset vs ideal circuit: max |diff| {err:.1e} (pulse compilation + 0.07 % reset error)")
    assert err < 5e-3
    print("check: frames, noiseless pulses, noisy loop and reset OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", default=",".join(LEVELS))
    parser.add_argument("--out", default="results/observable_sets/data/p3")
    parser.add_argument("--quick", action="store_true", help="smoke test: short series, 2 levels, 3 budgets")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
        raise SystemExit
    if args.check:
        check()
        raise SystemExit
    levels = ["sp-ideal", "sp-2q98"] if args.quick else args.levels.split(",")
    budgets = (1_000, 100_000, None) if args.quick else obs.BUDGETS
    length = 600 if args.quick else obs.LENGTH
    t0 = time.time()
    rows, feat_rows = run(args.n, args.seed, length, levels, budgets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / f"capacity_n{args.n}_seed{args.seed}.csv", index=False)
    feat_rows.to_csv(out / f"features_n{args.n}_seed{args.seed}.csv", index=False)
    print(f"done n{args.n}_seed{args.seed} in {time.time() - t0:.0f}s -> {out}", flush=True)
