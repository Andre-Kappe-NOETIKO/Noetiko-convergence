from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert, butter, filtfilt


@dataclass(frozen=True)
class PhaseExtractionResult:
    """Result of phase extraction.

    Attributes
    ----------
    phase : np.ndarray
        Instantaneous phase (radians), unwrapped if requested.
        Shape: (n_channels, n_samples)
    amplitude : np.ndarray
        Instantaneous analytic amplitude. Shape: (n_channels, n_samples)
    """
    phase: np.ndarray
    amplitude: np.ndarray


def bandpass_filtfilt(
    x: np.ndarray,
    fs_hz: float,
    band_hz: Tuple[float, float],
    order: int = 4,
) -> np.ndarray:
    """Zero-phase bandpass filter using Butterworth + filtfilt.

    Parameters
    ----------
    x : np.ndarray
        Array shaped (n_channels, n_samples) or (n_samples,).
    fs_hz : float
        Sampling rate in Hz.
    band_hz : (low, high)
        Bandpass in Hz. Must satisfy 0 < low < high < fs/2.
    order : int
        Butterworth filter order.

    Returns
    -------
    np.ndarray filtered signal with same shape as input.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x2 = x[None, :]
    elif x.ndim == 2:
        x2 = x
    else:
        raise ValueError("x must be 1D or 2D (channels x samples).")

    low, high = band_hz
    nyq = 0.5 * fs_hz
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid band {band_hz} for fs={fs_hz} (nyquist={nyq}).")

    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    y = filtfilt(b, a, x2, axis=-1)
    return y[0] if x.ndim == 1 else y


def hilbert_phase(
    x: np.ndarray,
    unwrap: bool = True,
    fs_hz: Optional[float] = None,
    band_hz: Optional[Tuple[float, float]] = None,
    filter_order: int = 4,
) -> PhaseExtractionResult:
    """Extract instantaneous phase using the analytic signal (Hilbert transform).

    Notes
    -----
    This is an operational phase map Φ used in Paper III. If band_hz is provided,
    the signal is bandpass filtered first (recommended for broadband data).

    Parameters
    ----------
    x : np.ndarray
        Signal array shaped (n_channels, n_samples) or (n_samples,).
    unwrap : bool
        Whether to unwrap the phase along time.
    fs_hz : float, optional
        Sampling rate. Required if band_hz is provided.
    band_hz : (low, high), optional
        Optional bandpass in Hz.
    filter_order : int
        Butterworth order if filtering.

    Returns
    -------
    PhaseExtractionResult
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x2 = x[None, :]
    elif x.ndim == 2:
        x2 = x
    else:
        raise ValueError("x must be 1D or 2D (channels x samples).")

    if band_hz is not None:
        if fs_hz is None:
            raise ValueError("fs_hz must be provided when band_hz is used.")
        x2 = bandpass_filtfilt(x2, fs_hz=fs_hz, band_hz=band_hz, order=filter_order)

    z = hilbert(x2, axis=-1)
    amp = np.abs(z)
    ph = np.angle(z)
    if unwrap:
        ph = np.unwrap(ph, axis=-1)

    return PhaseExtractionResult(phase=ph, amplitude=amp)
