from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QualityGateResult:
    passed: np.ndarray  # bool mask per window
    rejection_rate: float


def windowed_snr_gate(
    amplitude: np.ndarray,
    window_size: int,
    amp_threshold: float,
) -> QualityGateResult:
    """Amplitude/SNR proxy gate using analytic amplitude.

    Parameters
    ----------
    amplitude : np.ndarray
        Analytic amplitude shaped (n_channels, n_samples).
    window_size : int
        Window size in samples.
    amp_threshold : float
        Threshold on median amplitude (per window) to accept.

    Returns
    -------
    QualityGateResult:
        passed: bool array (n_windows,)
        rejection_rate: fraction rejected.
    """
    amp = np.asarray(amplitude, dtype=float)
    if amp.ndim != 2:
        raise ValueError("amplitude must be 2D (channels x samples).")
    n = amp.shape[-1]
    if window_size <= 0 or window_size > n:
        raise ValueError("Invalid window_size.")
    n_windows = n // window_size
    if n_windows < 1:
        raise ValueError("Not enough samples for one window.")

    passed = np.zeros(n_windows, dtype=bool)
    for w in range(n_windows):
        sl = slice(w*window_size, (w+1)*window_size)
        # median across time then channels
        m = np.median(amp[:, sl])
        passed[w] = bool(m >= amp_threshold)

    rej = 1.0 - float(np.mean(passed))
    return QualityGateResult(passed=passed, rejection_rate=rej)


def unwrap_consistency_gate(theta: np.ndarray, max_step: float = np.pi) -> bool:
    """Simple unwrap/aliasing check: max step between samples should be bounded."""
    theta = np.asarray(theta, dtype=float)
    d = np.diff(theta, axis=-1)
    return bool(np.all(np.abs(d) <= max_step))
