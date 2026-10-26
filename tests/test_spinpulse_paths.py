"""
SpinPulse backend: code paths the reference test does not reach.

1. Shot mode (precision > 0): sampled <Z>/<ZZ> must converge to the exact values, which
   checks the bitstring -> expectation conversion and its qubit order.
2. Noise-free runs for every encoding and for a Hamiltonian (TFIM) reservoir: the
   backend must reproduce an exact simulation of the same compiled circuit.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, PauliList

pytest.importorskip("spin_pulse")

from qrc.backends.numpy_backend import NumpyBackend
from qrc.backends.spinpulse import SpinPulseBackend, SpinPulseConfig
from qrc.encodings import AngleEncoding, DenseAngleEncoding, ReuploadingEncoding
from qrc.hamiltonians import tfim
from qrc.reservoirs import HamiltonianReservoir, RandomCircuitReservoir, Reservoir


def compiled_reservoir(backend, reservoir):
    """The reservoir as SpinPulse compiled it (noise free), as a plain one-gate reservoir.

    Comparing against this instead of the ideal circuit removes SpinPulse's small
    intrinsic gate error, so the backend's own wiring can be checked exactly.
    """
    unitary = Operator.from_circuit(backend._timeline.reservoir.to_circuit())

    class Compiled(Reservoir):
        def _build(self):
            qc = QuantumCircuit(self.num_qubits)
            qc.append(UnitaryGate(unitary), range(self.num_qubits))
            return qc

    return Compiled(reservoir.num_input_qubits, reservoir.num_mem_qubits)


# ------------------------------------------------------------------ 1. shot mode


def test_shots_converge_to_exact_values():
    rng = np.random.default_rng(0)
    windows = rng.random((6, 2, 2))
    encoder = AngleEncoding(2)
    reservoir = RandomCircuitReservoir(num_input_qubits=2, num_mem_qubits=1, depth=2, rng=1)
    # Z on each qubit and both ZZ pairs: a qubit-order mistake would swap or mix them.
    observables = PauliList(["IIZ", "IZI", "ZII", "IZZ", "ZZI"])

    exact = SpinPulseBackend(SpinPulseConfig(noise_type="none")).run_batch(
        windows, encoder, reservoir, observables
    )
    # Guard: the qubits must disagree, or a qubit-order bug could not show.
    assert np.ptp(exact, axis=1).max() > 0.2

    precision = 0.005  # 40,000 shots per window
    shots = SpinPulseBackend(SpinPulseConfig(noise_type="none", precision=precision, seed=1))
    sampled = shots.run_batch(windows, encoder, reservoir, observables)

    # Standard error of a +-1 average over N shots is sqrt((1 - <O>^2) / N); allow 5 of them.
    tolerance = 5 * np.sqrt((1 - exact**2) * precision**2) + 1e-9
    assert np.all(np.abs(sampled - exact) <= tolerance)


def test_shots_refuse_non_diagonal_observables():
    backend = SpinPulseBackend(SpinPulseConfig(noise_type="none", precision=0.1))
    windows = np.zeros((1, 1, 2))
    reservoir = RandomCircuitReservoir(num_input_qubits=2, num_mem_qubits=1, depth=1, rng=1)
    with pytest.raises(NotImplementedError, match="I/Z"):
        backend.run_batch(windows, AngleEncoding(2), reservoir, PauliList(["IIX"]))


# ------------------------------------------------- 2. every encoding, and a TFIM reservoir


CASES = {
    # name: (encoder, reservoir, number of features)
    "angle": (lambda: AngleEncoding(2), lambda: RandomCircuitReservoir(2, 1, depth=2, rng=1), 2),
    "dense": (lambda: DenseAngleEncoding(2), lambda: RandomCircuitReservoir(2, 1, depth=2, rng=1), 3),
    "reupload": (
        lambda: ReuploadingEncoding(2, num_layers=2, rng=3),
        lambda: RandomCircuitReservoir(2, 1, depth=2, rng=1),
        2,
    ),
    "tfim": (lambda: AngleEncoding(2), lambda: HamiltonianReservoir(tfim(3, rng=1), 2, steps=2), 2),
}


@pytest.mark.parametrize("case", list(CASES))
def test_noise_free_matches_exact_simulation(case):
    make_encoder, make_reservoir, num_features = CASES[case]
    encoder, reservoir = make_encoder(), make_reservoir()
    # Values outside [0, 1] give angles beyond pi, so the backend's angle wrapping is exercised.
    windows = np.random.default_rng(0).uniform(-0.2, 1.2, (8, 3, num_features))
    observables = PauliList(["IIZ", "IZI", "ZII", "IZZ", "ZZI", "IIX", "IYI"])

    backend = SpinPulseBackend(SpinPulseConfig(noise_type="none"))
    features = backend.run_batch(windows, encoder, reservoir, observables)

    expected = NumpyBackend().run_batch(
        windows, encoder, compiled_reservoir(backend, reservoir), observables
    )
    np.testing.assert_allclose(features, expected, atol=1e-10)