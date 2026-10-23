import numpy as np
import pytest

from qrc.baselines.esn import EchoStateNetwork


def test_esn_scales_recurrent_weights_to_spectral_radius():
    esn = EchoStateNetwork(input_size=3, n_units=20, spectral_radius=0.7, density=0.5, rng=0)

    assert np.max(np.abs(np.linalg.eigvals(esn.W))) == pytest.approx(0.7)


def test_esn_returns_final_state_per_window_and_is_seeded():
    windows = np.random.default_rng(0).random((5, 4, 3))

    states = EchoStateNetwork(input_size=3, n_units=10, rng=1).run(windows)

    assert states.shape == (5, 10)
    assert np.all(np.abs(states) < 1)
    assert np.array_equal(states, EchoStateNetwork(input_size=3, n_units=10, rng=1).run(windows))


def test_esn_state_depends_on_earlier_transactions_only_through_recurrence():
    rng = np.random.default_rng(0)
    windows = rng.random((2, 3, 2))
    windows[1, -1] = windows[0, -1]  # same last transaction, different history

    with_memory = EchoStateNetwork(input_size=2, n_units=6, spectral_radius=0.9, rng=0).run(windows)
    without_memory = EchoStateNetwork(input_size=2, n_units=6, spectral_radius=0.0, rng=0).run(windows)

    assert not np.allclose(with_memory[0], with_memory[1])
    assert np.allclose(without_memory[0], without_memory[1])


def test_esn_rejects_wrong_feature_count():
    with pytest.raises(ValueError):
        EchoStateNetwork(input_size=3, rng=0).run(np.zeros((2, 4, 2)))
