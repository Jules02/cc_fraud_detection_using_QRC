"""P2: memory capacity of observable sets under shot, gate and readout noise (input reset on).

One run = one (n qubits, seed). Reservoir: the CLI's TFIM (`--hamiltonian tfim`: H = sum J_ij X_i X_j + sum Z_i,
J_ij ~ U(0, 1) on a chain, evolution time 1, 2 Trotter steps), 1 input qubit with angle encoding RY(pi u), the
input qubit reset before every step. Task: short-term memory with i.i.d. uniform inputs, target u_{t-d} for
d = 0..L-1, memory capacity MC = sum_d C(d), C(d) = corr(y_hat, y)^2 on the test windows, ridge readout with alpha
chosen per delay on the last 20 % of the training windows.

Noise (NOISE). Density matrices are simulated gate by gate (the reservoir transpiled to rx/ry/rz/rzz, the native
spin-qubit gate set). Every gate is followed by a depolarizing channel on its qubits whose average gate infidelity
is the level's 1q / 2q infidelity (lambda = infidelity * d / (d - 1)), so a Pauli string on the gate's qubits
shrinks by (1 - lambda). The input reset prepares |1> with the readout flip probability. Measurement: the hardware
only reads Z, so a setting is (frame unitary V, then Z on every qubit); qubits measured in X/Y or a random frame
get one extra noisy 1q gate before readout, and every outcome bit flips with the readout probability (no
mitigation).

Observable sets (pauli_settings); a setting gives every Z-type string of its frame for free:
- Z-w1: local Z (n features), Z-strings: every Z string (2^n - 1), both 1 setting;
- frames-3: Z-strings in the Z frame and in 2 fixed Haar-random single-qubit frames (3 settings);
- XYZ-strings: every single-letter string X.., Y.., Z.. (3 settings);
- XYZ+XYZ-2body: local X/Y/Z plus the 9 nearest-neighbour two-body products (9 settings);
- all-Pauli: full tomography, 3^n settings (n = 4 only).

Shots: a budget of B shots per window is split evenly over the settings, multinomial sampling per window and
setting (features of one setting share the counts), train and test both sampled. Every (level, set, budget) cell
draws from its own seeded stream, so a cell's numbers do not depend on which other cells are run.

Usage:
    python results/observable_sets/observables.py --check
    python results/observable_sets/observables.py --n 5 --seed 42 --out results/observable_sets/data/p2
    python results/observable_sets/observables.py --n 5 --seed 42 --quick   (smoke test)
"""
import argparse
import itertools
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Pauli, PauliList, random_unitary

from qrc.backends.numpy_backend import NumpyBackend
from qrc.benchmarks.windows import sliding_windows
from qrc.encodings import AngleEncoding
from qrc.hamiltonians import tfim
from qrc.observables import build_observables
from qrc.protocol import QRCProtocol
from qrc.reservoirs import HamiltonianReservoir

L = 10
LENGTH = 5000
TEST_SPLIT = 0.2
BUDGETS = (100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 1_000_000, None)  # shots per window, None = exact
ALPHAS = np.logspace(-6, 4, 11)
MAX_PAULIS = 255  # all-Pauli only where 4^n - 1 <= this (n = 4)
NATIVE = ["rx", "ry", "rz", "rzz"]
SETS = ("Z-w1", "Z-strings", "frames-3", "XYZ-strings", "XYZ+XYZ-2body", "all-Pauli")

# name -> (2q gate infidelity, 1q gate infidelity, readout flip probability). "2q2" is Quobly's current device
# (98 % 2q, 99.6 % 1q, 99.93 % readout); "2q2-ro1" is the same with 1 % readout error.
NOISE = {
    "ideal": (0.0, 0.0, 0.0),
    "2q0.5": (0.005, 0.001, 0.0007),
    "2q1": (0.01, 0.002, 0.0007),
    "2q2": (0.02, 0.004, 0.0007),
    "2q2-ro1": (0.02, 0.004, 0.01),
}

LETTERS = string.ascii_lowercase


def cell_rng(seed, n, level_id, name, budget):
    """Independent stream per (seed, n, noise level, observable set, budget)."""
    b = len(BUDGETS) - 1 if budget is None else BUDGETS.index(budget)
    return np.random.default_rng([seed, n, level_id, SETS.index(name), b])


# ---------------------------------------------------------------- density-matrix simulation

def _axis(q, n):
    """Tensor axis of qubit q (Qiskit little-endian: qubit 0 is the last axis)."""
    return n - 1 - q


def apply_gate(rho, G, qubits, n):
    """rho -> G rho G^dagger for a k-qubit gate on `qubits` (G little-endian in its qargs). G is (2^k, 2^k) or
    batched (M, 2^k, 2^k)."""
    M, D = rho.shape[:2]
    k = len(qubits)
    batched = G.ndim == 3
    t = rho.reshape((M,) + (2,) * (2 * n))
    row, col = list(LETTERS[:n]), list(LETTERS[n:2 * n])
    new = LETTERS[2 * n:2 * n + k]
    Gt = G.reshape(((M,) if batched else ()) + (2,) * (2 * k))
    ax = [_axis(qubits[k - 1 - j], n) for j in range(k)]  # gate axis j <-> qubit qubits[k-1-j]
    g_in = "".join(row[a] for a in ax)
    g_out = "".join(new)
    b = "Z" if batched else ""
    out_row = row.copy()
    for j, a in enumerate(ax):
        out_row[a] = new[j]
    t = np.einsum(f"{b}{g_out}{g_in},Z{''.join(row)}{''.join(col)}->Z{''.join(out_row)}{''.join(col)}", Gt, t,
                  optimize=True)
    c_in = "".join(col[a] for a in ax)
    out_col = col.copy()
    for j, a in enumerate(ax):
        out_col[a] = new[j]
    t = np.einsum(f"{b}{g_out}{c_in},Z{''.join(row)}{''.join(col)}->Z{''.join(row)}{''.join(out_col)}",
                  Gt.conj(), t, optimize=True)
    return t.reshape(M, D, D)


def depolarize(rho, qubits, lam, n):
    """rho -> (1 - lam) rho + lam (I / 2^k) (x) Tr_qubits(rho)."""
    if lam == 0 or not qubits:
        return rho
    M, D = rho.shape[:2]
    t = rho.reshape((M,) + (2,) * (2 * n))
    row, col = list(LETTERS[:n]), list(LETTERS[n:2 * n])
    ax = [_axis(q, n) for q in qubits]
    col_in = [row[i] if i in ax else col[i] for i in range(n)]
    keep = "".join(row[i] for i in range(n) if i not in ax) + "".join(col[i] for i in range(n) if i not in ax)
    red = np.einsum(f"Z{''.join(row)}{''.join(col_in)}->Z{keep}", t)
    subs = [f"Z{keep}"] + [row[a] + col[a] for a in ax]
    full = np.einsum(",".join(subs) + f"->Z{''.join(row)}{''.join(col)}", red, *[np.eye(2) / 2] * len(ax))
    return ((1 - lam) * t + lam * full).reshape(M, D, D)


def lam_of(infidelity, k):
    d = 2 ** k
    return infidelity * d / (d - 1)


def reservoir_ops(reservoir, n):
    """Native gate list [(matrix, qubits)] of the reservoir circuit."""
    qc = transpile(reservoir.circuit(), basis_gates=NATIVE, optimization_level=1)
    assert Operator(qc).equiv(Operator(reservoir.circuit())), "transpiled reservoir differs"
    return [(Operator(inst.operation).data, tuple(qc.find_bit(q).index for q in inst.qubits)) for inst in qc.data]


def apply_ops(rho, ops, noise, n):
    p2, p1, _ = noise
    for G, qubits in ops:
        rho = apply_gate(rho, G, qubits, n)
        rho = depolarize(rho, qubits, lam_of(p2 if len(qubits) == 2 else p1, len(qubits)), n)
    return rho


def ry_batch(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def density_matrices(windows, ops, n, noise):
    """Final state of every window, (M, 2^n, 2^n). Qubit 0 is the input qubit, reset before every step after the
    first (prepared in |1> with the readout flip probability)."""
    p2, p1, p_ro = noise
    D = 2 ** n
    M = windows.shape[0]
    rho = np.zeros((M, D, D), complex)
    rho[:, 0, 0] = 1
    for t in range(windows.shape[1]):
        if t > 0:
            r = rho.reshape(M, D // 2, 2, D // 2, 2)
            tr = r[:, :, 0, :, 0] + r[:, :, 1, :, 1]
            rho = np.zeros_like(rho)
            v = rho.reshape(M, D // 2, 2, D // 2, 2)
            v[:, :, 0, :, 0] = (1 - p_ro) * tr
            v[:, :, 1, :, 1] = p_ro * tr
        rho = apply_gate(rho, ry_batch(np.pi * windows[:, t, 0]), (0,), n)
        rho = depolarize(rho, (0,), lam_of(p1, 1), n)
        rho = apply_ops(rho, ops, noise, n)
    return rho


# ---------------------------------------------------------------- measurement settings

def strings(letter, n, max_w=None):
    out = []
    for m in range(1, 2 ** n):
        w = bin(m).count("1")
        if max_w is None or w <= max_w:
            out.append(Pauli("".join(letter if (m >> (n - 1 - q)) & 1 else "I" for q in range(n))))
    return PauliList(out)


def basis_frame(basis):
    """(V, rotated qubits) mapping a Pauli basis string (little-endian) to Z: H for X, H S^dagger for Y."""
    qc = QuantumCircuit(len(basis))
    rotated = []
    for q, b in enumerate(reversed(basis)):
        if b == "X":
            qc.h(q)
        elif b == "Y":
            qc.sdg(q)
            qc.h(q)
        if b != "Z":
            rotated.append(q)
    return Operator(qc).data, tuple(rotated)


def qwc_settings(paulis):
    """Qubit-wise-commuting grouping -> [(key, V, rotated qubits, group)]."""
    out = []
    for group in paulis.group_qubit_wise_commuting():
        basis = ["Z"] * paulis.num_qubits
        for p in group:
            for q in range(paulis.num_qubits):
                if p.x[q] or p.z[q]:
                    basis[-(q + 1)] = str(p[q])
        basis = "".join(basis)
        V, rotated = basis_frame(basis)
        out.append((basis, V, rotated, group))
    return out


def pauli_settings(n, seed):
    """Observable set name -> list of settings (key, V, rotated qubits, group), in SETS order. `group` lists the
    Paulis read in that setting, written in its measurement frame."""
    I = np.eye(2 ** n)
    everything = tuple(range(n))
    z_all = strings("Z", n)
    sets = {"Z-w1": [("Z", I, (), strings("Z", n, 1))], "Z-strings": [("Z", I, (), z_all)]}
    rng = np.random.default_rng(10_000 + seed)
    frames = []
    for i in range(2):
        V = np.array([[1.0]])
        for _ in range(n):  # kron builds qubit n-1 first = most significant
            V = np.kron(V, random_unitary(2, seed=rng).data)
        frames.append((f"frame{i}", V, everything, z_all))
    sets["frames-3"] = sets["Z-strings"] + frames
    sets["XYZ-strings"] = qwc_settings(PauliList(list(strings("X", n)) + list(strings("Y", n)) + list(z_all)))
    sets["XYZ+XYZ-2body"] = qwc_settings(build_observables("XYZ+XYZ-2body", n))
    if 4 ** n - 1 <= MAX_PAULIS:
        sets["all-Pauli"] = qwc_settings(PauliList(["".join(p) for p in itertools.product("IXYZ", repeat=n)][1:]))
    return sets


def outcome_probs(rho, V, rotated, noise, n):
    """Z-basis outcome distribution after the frame change (one noisy 1q gate per rotated qubit) and readout
    flips, (M, 2^n)."""
    _, p1, p_ro = noise
    for q in rotated:
        rho = depolarize(rho, (q,), lam_of(p1, 1), n)
    return readout(np.einsum("xa,mab,xb->mx", V, rho, V.conj()).real, p_ro, n)


def readout(p, p_ro, n):
    """Flip every outcome bit independently with probability p_ro, then renormalize. p is (M, 2^n)."""
    if p_ro > 0:
        flip = np.array([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]])
        t = p.reshape((-1,) + (2,) * n)
        for a in range(n):
            t = np.moveaxis(np.tensordot(t, flip, axes=([a + 1], [0])), -1, a + 1)
        p = t.reshape(p.shape)
    p = np.clip(p, 0, None)
    return p / p.sum(1, keepdims=True)


def eigenvalues(group, n):
    """+-1 value of every (frame-)Z-type string of the group on every outcome bitstring, (2^n, len(group))."""
    x = np.arange(2 ** n)
    masks = [sum(1 << q for q in range(n) if p.x[q] or p.z[q]) for p in group]
    return np.stack([1 - 2 * (np.bitwise_count(x & m).astype(int) & 1) for m in masks], 1).astype(float)


def features(probs, settings, n, shots, rng):
    cols = []
    for key, _, _, group in settings:
        p = probs[key]
        if shots is not None:
            p = rng.multinomial(shots, p) / shots
        cols.append(p @ eigenvalues(group, n))
    return np.concatenate(cols, 1)


def weights(settings):
    return np.array([int(np.sum(p.x | p.z)) for _, _, _, group in settings for p in group])


# ---------------------------------------------------------------- readout

def ridge_capacity(X_train, Y_train, X_test, Y_test):
    """Per-target capacity corr(y_hat, y)^2, alpha chosen per target on the last 20 % of the training set."""
    def fit(X, Y):
        mu, sd = X.mean(0), X.std(0)
        sd[sd < 1e-9] = np.inf  # a constant column would otherwise standardize rounding noise to O(1)
        Z = (X - mu) / sd
        try:
            U_, S, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:  # OpenBLAS's gesdd occasionally fails to converge; gesvd is slower but robust
            print(f"  svd fallback to gesvd: shape {Z.shape}", flush=True)
            U_, S, Vt = scipy.linalg.svd(Z, full_matrices=False, lapack_driver="gesvd")
        UtY = U_.T @ (Y - Y.mean(0))
        return [(mu, sd, Vt.T @ ((S / (S ** 2 + a))[:, None] * UtY), Y.mean(0)) for a in ALPHAS]

    def predict(model, X):
        mu, sd, W, b = model
        return (X - mu) / sd @ W + b

    def corr2(P, Y):
        P, Y = P - P.mean(0), Y - Y.mean(0)
        den = np.sqrt((P ** 2).sum(0) * (Y ** 2).sum(0))
        return np.where(den > 0, (P * Y).sum(0) / np.where(den > 0, den, 1), 0) ** 2

    cut = int(0.8 * len(X_train))
    val = np.stack([corr2(predict(m, X_train[cut:]), Y_train[cut:]) for m in fit(X_train[:cut], Y_train[:cut])])
    best = val.argmax(0)
    models = fit(X_train, Y_train)
    C = np.array([corr2(predict(models[a], X_test)[:, [d]], Y_test[:, [d]])[0] for d, a in enumerate(best)])
    return C, ALPHAS[best]


# ---------------------------------------------------------------- task and run

def task_data(seed, length):
    """Short-term memory windows (M, L, 1) and targets (M, L), column d = u_{t-d}, split in time order."""
    u = np.random.default_rng([seed, 0]).uniform(0, 1, length)
    X = sliding_windows(u, L)
    Y = X[:, ::-1]
    cut = int(len(X) * (1 - TEST_SPLIT))
    return X[:cut, :, None], Y[:cut], X[cut:, :, None], Y[cut:]


def make_reservoir(n, seed):
    """The CLI's `--hamiltonian tfim` reservoir (time 1, 2 Trotter steps)."""
    return HamiltonianReservoir(tfim(n, rng=seed), num_input_qubits=1, time=1.0, steps=2)


def evaluate(probs, sets, n, seed, level, level_id, budgets, Y_train, Y_test):
    """Capacity rows and per-weight feature-spread rows for every (set, budget), from the outcome distributions.
    `level_id` (an integer unique to the noise level) seeds the per-cell sampling streams."""
    rows, feat_rows = [], []
    for name, settings in sets.items():
        n_settings = len(settings)
        exact = features(probs["train"], settings, n, None, None)
        w = weights(settings)
        for weight in np.unique(w):
            feat_rows.append(dict(n=n, seed=seed, noise=level, observables=name, weight=int(weight),
                                  n_features=int((w == weight).sum()),
                                  median_std=float(np.median(exact[:, w == weight].std(0)))))
        for budget in budgets:
            shots = None if budget is None else budget // n_settings
            if shots == 0:
                continue
            rng = cell_rng(seed, n, level_id, name, budget)
            F_train = features(probs["train"], settings, n, shots, rng)
            F_test = features(probs["test"], settings, n, shots, rng)
            C, alphas = ridge_capacity(F_train, Y_train, F_test, Y_test)
            chance, _ = ridge_capacity(F_train, Y_train, F_test, rng.permutation(Y_test))
            for d in range(Y_train.shape[1]):
                rows.append(dict(n=n, seed=seed, noise=level, observables=name, n_features=F_train.shape[1],
                                 n_settings=n_settings, budget=np.inf if budget is None else budget,
                                 shots_per_setting=shots, delay=d, capacity=C[d], chance=chance[d], alpha=alphas[d]))
    return rows, feat_rows


def run(n, seed, length, levels, budgets):
    ops = reservoir_ops(make_reservoir(n, seed), n)
    sets = pauli_settings(n, seed)
    keys = {key: (V, rot) for s in sets.values() for key, V, rot, _ in s}
    X_train, Y_train, X_test, Y_test = task_data(seed, length)
    rows, feat_rows = [], []
    for level in levels:
        t0 = time.time()
        noise = NOISE[level]
        probs = {}
        for split, X in (("train", X_train), ("test", X_test)):
            rho = density_matrices(X, ops, n, noise)
            probs[split] = {key: outcome_probs(rho, V, rot, noise, n) for key, (V, rot) in keys.items()}
        r, f = evaluate(probs, sets, n, seed, level, list(NOISE).index(level), budgets, Y_train, Y_test)
        rows += r
        feat_rows += f
        print(f"n={n} seed={seed} {level}: {time.time() - t0:.1f}s", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(feat_rows)


def check():
    """Noiseless gate-by-gate density matrices vs NumpyBackend (reset on), outcome distributions vs Tr(P rho), and
    the depolarizing channel's action on Paulis."""
    X = task_data(42, 60)[0][:20]
    for n in (4, 5):
        reservoir = make_reservoir(n, 42)
        obs = build_observables("XYZ+XYZ-2body", n)
        ref = QRCProtocol(AngleEncoding(num_qubits=1), reservoir, obs, NumpyBackend(reset_inputs=True)).run(X)
        rho = density_matrices(X, reservoir_ops(reservoir, n), n, NOISE["ideal"])
        err = np.abs(np.einsum("pab,mba->mp", obs.to_matrix(array=True), rho).real - ref).max()
        assert err < 1e-8, f"density matrices disagree with NumpyBackend (n={n}): {err:.2e}"
        for name, settings in pauli_settings(n, 42).items():
            probs = {key: outcome_probs(rho, V, rot, NOISE["ideal"], n) for key, V, rot, _ in settings}

            def observable(V, group):  # QWC groups hold Paulis; frame groups hold Z strings P, observable V^† P V
                P = group.to_matrix(array=True)
                return P if group.x.any() else np.einsum("xa,pxy,yb->pab", V.conj(), P, V)

            direct = np.concatenate([np.einsum("pab,mba->mp", observable(V, group), rho).real
                                     for _, V, _, group in settings], 1)
            err = np.abs(features(probs, settings, n, None, None) - direct).max()
            assert err < 1e-8, f"outcome distributions disagree with Tr(P rho) for {name}: {err:.2e}"
    rho = density_matrices(X[:5], reservoir_ops(make_reservoir(4, 42), 4), 4, NOISE["ideal"])
    lam = 0.1
    noisy = depolarize(rho, (1, 2), lam, 4)  # shrinks Paulis touching qubits 1, 2 by (1 - lam), leaves others
    for label in ("IXYI", "ZXII", "IIIZ"):
        P = Pauli(label).to_matrix()
        before = np.einsum("ab,mba->m", P, rho).real
        after = np.einsum("ab,mba->m", P, noisy).real
        assert np.allclose(after, before * ((1 - lam) if label != "IIIZ" else 1)), f"depolarize wrong on {label}"
    assert np.allclose(np.trace(noisy, axis1=1, axis2=2), 1)
    print("check: gate-level simulation matches NumpyBackend, settings match Tr(P rho), depolarizing OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="run the validation checks and exit")
    parser.add_argument("--n", type=int, default=5, help="total qubits (1 input + n-1 memory)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--levels", default=",".join(NOISE), help="comma-separated noise levels")
    parser.add_argument("--out", default="results/observable_sets/data/p2")
    parser.add_argument("--quick", action="store_true", help="smoke test: short series, 2 levels, 3 budgets")
    args = parser.parse_args()

    if args.check:
        check()
        raise SystemExit
    levels = ["ideal", "2q2"] if args.quick else args.levels.split(",")
    budgets = (1_000, 100_000, None) if args.quick else BUDGETS
    length = 600 if args.quick else LENGTH
    t0 = time.time()
    rows, feat_rows = run(args.n, args.seed, length, levels, budgets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / f"capacity_n{args.n}_seed{args.seed}.csv", index=False)
    feat_rows.to_csv(out / f"features_n{args.n}_seed{args.seed}.csv", index=False)
    print(f"done n{args.n}_seed{args.seed} in {time.time() - t0:.0f}s -> {out}", flush=True)
