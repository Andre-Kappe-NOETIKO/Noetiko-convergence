from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def phase_randomization_surrogate(
    x: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Fourier phase randomization surrogate (preserves power spectrum).

    Parameters
    ----------
    x : np.ndarray
        Signal shaped (n_channels, n_samples) or (n_samples,).
    rng : np.random.Generator, optional
        Random generator.

    Returns
    -------
    np.ndarray surrogate signal with same shape as x.

    Notes
    -----
    This is a standard surrogate used to test whether coordination signatures
    exceed what is explainable by marginal spectra alone.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x2 = x[None, :]
    elif x.ndim == 2:
        x2 = x
    else:
        raise ValueError("x must be 1D or 2D (channels x samples).")

    n = x2.shape[-1]
    X = np.fft.rfft(x2, axis=-1)
    mag = np.abs(X)

    # Randomize phases except DC and Nyquist (if present)
    rand_ph = rng.uniform(0, 2*np.pi, size=X.shape)
    rand_ph[..., 0] = 0.0
    if n % 2 == 0:
        rand_ph[..., -1] = 0.0

    Y = mag * np.exp(1j * rand_ph)
    y = np.fft.irfft(Y, n=n, axis=-1)
    return y[0] if x.ndim == 1 else y


def time_shift_surrogate(
    x: np.ndarray,
    min_shift: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Circular time-shift surrogate across channels.

    Parameters
    ----------
    x : np.ndarray
        Signal shaped (n_channels, n_samples).
    min_shift : int
        Minimum circular shift in samples (to exceed coherence time).
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray shifted signal with same shape.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must be 2D (channels x samples) for time_shift_surrogate.")
    n_ch, n = x.shape
    if min_shift < 1 or min_shift >= n:
        raise ValueError("min_shift must satisfy 1 <= min_shift < n_samples.")

    y = np.empty_like(x)
    for i in range(n_ch):
        shift = int(rng.integers(min_shift, n))
        y[i] = np.roll(x[i], shift)
    return y
