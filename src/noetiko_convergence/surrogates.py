from __future__ import annotations

from typing import Optional

import numpy as np


def time_shift_surrogate(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Circular time shift surrogate.
    """
    x = np.asarray(x)
    shift = int(shift) % x.shape[0]
    return np.roll(x, shift=shift, axis=0)


def fourier_phase_randomization(
    x: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Fourier phase randomization surrogate (preserves power spectrum approximately).

    For real-valued signals, we randomize phases of positive frequencies and
    reconstruct a real signal via inverse FFT.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1D for this surrogate")

    if rng is None:
        rng = np.random.default_rng()

    n = x.shape[0]
    X = np.fft.rfft(x)
    mag = np.abs(X)
    phase = np.angle(X)

    # randomize phases except DC (0) and Nyquist (if present)
    rand = rng.uniform(0.0, 2.0 * np.pi, size=phase.shape)
    rand[0] = phase[0]
    if n % 2 == 0:
        rand[-1] = phase[-1]

    Xs = mag * np.exp(1j * rand)
    xs = np.fft.irfft(Xs, n=n)
    return xs


def phase_randomization_multichannel(
    X: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply Fourier phase randomization independently to each channel.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (T, N)")

    if rng is None:
        rng = np.random.default_rng()

    T, N = X.shape
    out = np.empty((T, N), dtype=float)
    for j in range(N):
        out[:, j] = fourier_phase_randomization(X[:, j], rng=rng)
    return out
