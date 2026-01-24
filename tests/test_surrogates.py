import numpy as np

from noetiko_convergence.surrogates import fourier_phase_randomization, time_shift_surrogate


def test_time_shift_surrogate_preserves_shape():
    x = np.arange(10)
    y = time_shift_surrogate(x, shift=3)
    assert y.shape == x.shape


def test_fourier_phase_randomization_preserves_length():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)
    y = fourier_phase_randomization(x, rng=rng)
    assert y.shape == x.shape
