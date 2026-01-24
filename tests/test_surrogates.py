import numpy as np
from noetiko_convergence.surrogates import phase_randomization_surrogate, time_shift_surrogate


def test_phase_randomization_preserves_power():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2048)
    y = phase_randomization_surrogate(x, rng=rng)

    X = np.abs(np.fft.rfft(x))
    Y = np.abs(np.fft.rfft(y))
    # spectra should match closely (numerical tolerance)
    assert np.allclose(X, Y, rtol=1e-5, atol=1e-5)


def test_time_shift_changes_alignment():
    rng = np.random.default_rng(0)
    n = 2000
    t = np.arange(n)
    x1 = np.sin(2*np.pi*0.01*t)
    x = np.vstack([x1, x1.copy()])

    y = time_shift_surrogate(x, min_shift=200, rng=rng)
    # channel 0 and 1 should no longer be perfectly aligned
    corr = np.corrcoef(y[0], y[1])[0, 1]
    assert corr < 0.95
