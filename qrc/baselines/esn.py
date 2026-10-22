"""Echo state network: the classical counterpart of the quantum reservoir.

A fixed random recurrent network replaces the quantum circuit; only the
logistic readout is trained, exactly as for QRC. Each window starts from a zero
state, like the reset-window QRC, and the state after its last transaction is
the feature vector.
"""

import numpy as np


class EchoStateNetwork:
    """Leaky tanh reservoir ``x_t = (1 - a) x_{t-1} + a tanh(W_in u_t + W x_{t-1} + b)``.

    Args:
        input_size: features per transaction.
        n_units: reservoir neurons (the readout's input width).
        spectral_radius: largest absolute eigenvalue of ``W``; below 1 for fading memory.
        leak_rate: ``a`` in (0, 1]; smaller means slower state updates, longer memory.
        input_scaling: ``W_in`` entries are uniform in ``[-input_scaling, input_scaling]``.
        bias_scaling: ``b`` entries are uniform in ``[-bias_scaling, bias_scaling]``.
        density: fraction of non-zero entries in ``W``.
        rng: seed or numpy Generator.
    """

    def __init__(
        self,
        input_size,
        n_units=50,
        spectral_radius=0.9,
        leak_rate=1.0,
        input_scaling=1.0,
        bias_scaling=0.2,
        density=1.0,
        rng=None,
    ):
        if input_size < 1 or n_units < 1:
            raise ValueError("input_size and n_units must be positive")
        if not 0 < leak_rate <= 1:
            raise ValueError("leak_rate must be in (0, 1]")
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")
        if spectral_radius < 0:
            raise ValueError("spectral_radius must be non-negative")
        rng = np.random.default_rng(rng)

        self.n_units = n_units
        self.leak_rate = leak_rate
        self.W_in = rng.uniform(-input_scaling, input_scaling, size=(input_size, n_units))
        self.bias = rng.uniform(-bias_scaling, bias_scaling, size=n_units)
        W = rng.uniform(-1, 1, size=(n_units, n_units))
        W *= rng.random((n_units, n_units)) < density
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        self.W = W * (spectral_radius / radius) if radius > 0 else W

    def run(self, windows):
        """Map windows ``(samples, window_length, features)`` to final states ``(samples, n_units)``."""
        windows = np.asarray(windows, dtype=float)
        if windows.ndim != 3:
            raise ValueError("windows must have shape (samples, window_length, features)")
        if windows.shape[2] != self.W_in.shape[0]:
            raise ValueError(
                f"expected {self.W_in.shape[0]} features per transaction, got {windows.shape[2]}"
            )
        state = np.zeros((len(windows), self.n_units))
        for t in range(windows.shape[1]):
            update = np.tanh(windows[:, t] @ self.W_in + state @ self.W + self.bias)
            state = (1 - self.leak_rate) * state + self.leak_rate * update
        return state
