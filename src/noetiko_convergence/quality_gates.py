from __future__ import annotations

import numpy as np


def spectral_gate(X: np.ndarray, fs: float, f_lo: float, f_hi: float) -> bool:
    """
    Minimal spectral gate: checks if there is non-trivial power in a band.
    This is intentionally conservative and should be refined per dataset.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be (T, N)")

    T, _ = X.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return False

    P = np.abs(np.fft.rfft(X, axis=0)) ** 2
    band_power = float(np.mean(P[band, :]))
    total_power = float(np.mean(P))
    if total_power <= 0.0:
        return False

    return (band_power / total_power) > 0.05


def robustness_gate(phases: np.ndarray, amp: np.ndarray | None = None) -> bool:
    """
    Minimal robustness gate placeholder.
    Currently checks phase variance is non-degenerate; amplitude can be used for SNR thresholds.
    """
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 2:
        raise ValueError("phases must be (T, N)")

    v = float(np.var(phases))
    if not np.isfinite(v):
        return False

    if amp is not None:
        amp = np.asarray(amp, dtype=float)
        if float(np.mean(amp)) <= 1e-6:
            return False

    return v > 1e-6


def consistency_gate(phases: np.ndarray, max_step: float = 5.0) -> bool:
    """
    Minimal consistency gate: checks that unwrapped phase increments are not implausibly large.
    """
    phases = np.asarray(phases, dtype=float)
    if phases.ndim != 2:
        raise ValueError("phases must be (T, N)")

    d = np.diff(phases, axis=0)
    m = float(np.max(np.abs(d)))
    if not np.isfinite(m):
        return False

    return m < max_step
