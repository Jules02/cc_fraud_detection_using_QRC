"""Reference tests for the SpinPulse backend.
With noise, the backend must reproduce an independent, slow computation that builds
every block with SpinPulse's own classes (PulseLayer, PulseCircuit,
attach_time_traces, to_circuit) from the same noise traces.

With reset_inputs, the sampled (trajectory) reset must agree with the exact
density-matrix reset of EstimatorBackend, within the backend's reported error.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, PauliList, Statevector

pytest.importorskip("spin_pulse")

from spin_pulse import ExperimentalEnvironment, PulseCircuit
from spin_pulse.environment.noise import NoiseType
from spin_pulse.transpilation.instructions import IdleInstruction
from spin_pulse.transpilation.pulse_layer import PulseLayer
from spin_pulse.transpilation.pulse_sequence import PulseSequence

from qrc.backends.estimator import EstimatorBackend
from qrc.backends.spinpulse import SpinPulseBackend, SpinPulseConfig
from qrc.encodings import AngleEncoding

# 3 windows x 2 timesteps x 1 feature. Angles = pi * value stay inside (0, pi), so the
# backend's angle wrapping never changes them.
WINDOWS = np.array([[[0.1], [0.8]], [[0.5], [0.3]], [[0.9], [0.05]]])
# X is included on purpose: exact mode (precision=0) supports any Pauli
OBSERVABLES = PauliList(["IZ", "ZI", "ZZ", "XI"])


class TinyReservoir:
    """
    1 input + 1 memory qubit with an exchange (rzz) gate: small but not trivial.
    """

    num_input_qubits = 1
    num_qubits = 2

    def circuit(self):
        qc = QuantumCircuit(2)
        qc.rzz(0.7, 0, 1)
        qc.rx(0.4, 1)
        qc.ry(0.3, 0)
        return qc


def spinpulse_reference(backend, windows, config):
    """
    Slow reference: every block built and propagated by SpinPulse itself.

    Same timeline as the backend: per timestep one encoding layer (the rotation on the
    input qubit, padded to the pulse length of angle pi; the memory qubit idles), then
    the reservoir. One realization = one set of noise traces covering the window.
    """
    specs = backend._timeline.specs  # same seeded compilation as the backend
    frame = QuantumCircuit(2)
    qubits = frame.qubits

    def block_unitary(pulse, env, t_start):
        pulse.t_lab = t_start
        pulse.attach_time_traces(env)
        return Operator.from_circuit(pulse.to_circuit())

    def encoding_unitary(angle, env, t_start):
        rotation = specs.rotation_generator.from_angle("y", [qubits[0]], np.pi, specs)
        rotation.amplitude *= angle / np.pi  # same pulse length for every angle
        idle = IdleInstruction([qubits[1]], duration=rotation.duration)
        layer = PulseLayer(qubits, [PulseSequence([rotation]), PulseSequence([idle])], [])
        return block_unitary(PulseCircuit(frame, qubits, [layer], specs), env, t_start)

    enc_duration = specs.rotation_generator.from_angle("y", [qubits[0]], np.pi, specs).duration
    reservoir = PulseCircuit.from_circuit(
        specs.gate_transpile(TinyReservoir().circuit()), hardware_specs=specs
    )
    step = enc_duration + reservoir.duration
    window_duration = windows.shape[1] * step
    duration = window_duration + window_duration % 2
    env = ExperimentalEnvironment(
        hardware_specs=specs,
        noise_type=NoiseType[config.noise_type.upper()],
        T2S=config.T2S,
        TJS=config.TJS,
        duration=duration,
        only_idle=config.only_idle,
        segment_duration=1 if config.noise_type == "white" else duration,
        seed=config.seed,
    )

    total = np.zeros((len(windows), len(OBSERVABLES)))
    for m in range(config.noise_samples):
        if m > 0:
            env.generate_time_traces()  # fresh traces per realization, as in the backend
        for w, window in enumerate(windows):
            state = Statevector.from_label("00")
            for t, (value,) in enumerate(window):
                start = t * step
                state = state.evolve(encoding_unitary(np.pi * value, env, start))
                state = state.evolve(block_unitary(reservoir, env, start + enc_duration))
            total[w] += [state.expectation_value(p).real for p in OBSERVABLES]
    return total / config.noise_samples


@pytest.mark.parametrize("dd", ["none", "spin-echo"])
@pytest.mark.parametrize("noise_type", ["white", "pink", "quasistatic"])
def test_noisy_backend_matches_spinpulse_reference(noise_type, dd):
    config = SpinPulseConfig(
        noise_type=noise_type,
        T2S=100.0,
        dynamical_decoupling=dd,
        noise_samples=4,
        seed=11,
    )
    backend = SpinPulseBackend(config)
    features = backend.run_batch(WINDOWS, AngleEncoding(1), TinyReservoir(), OBSERVABLES)

    noiseless = SpinPulseBackend(SpinPulseConfig(noise_type="none")).run_batch(
        WINDOWS, AngleEncoding(1), TinyReservoir(), OBSERVABLES
    )
    # Guard: the noise must actually change the result, or the comparison proves nothing.
    assert not np.allclose(features, noiseless, atol=1e-3)

    expected = spinpulse_reference(backend, WINDOWS, config)
    np.testing.assert_allclose(features, expected, atol=1e-10)


def test_reset_matches_exact_density_matrix():
    observables = PauliList(["IZ", "ZI", "ZZ"])
    backend = SpinPulseBackend(
        SpinPulseConfig(noise_type="none", noise_samples=4000, seed=3),
        reset_inputs=True,
    )
    features = backend.run_batch(WINDOWS, AngleEncoding(1), TinyReservoir(), observables)

    # Exact reference: Aer's density-matrix reset, run on SpinPulse's own noise-free
    # compiled reservoir, so the hardware's intrinsic gate error is the same on both sides.
    compiled = Operator.from_circuit(backend._timeline.reservoir.to_circuit())

    class CompiledReservoir(TinyReservoir):
        def circuit(self):
            qc = QuantumCircuit(2)
            qc.append(UnitaryGate(compiled), [0, 1])
            return qc

    exact = EstimatorBackend(reset_inputs=True).run_batch(
        WINDOWS, AngleEncoding(1), CompiledReservoir(), observables
    )

    # The backend reports its own statistical error; allow 5 of those.
    tolerance = 5 * backend.last_stderr + 1e-9
    assert np.all(np.abs(features - exact) <= tolerance)