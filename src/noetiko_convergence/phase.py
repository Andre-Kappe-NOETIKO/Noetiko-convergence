from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.signal import hilbert


def analytic_signal_phase(x: np.ndarray) -> np.ndarray:
    """
    Extract instantaneous phase via analytic signal (Hilbert transform).

    Args:
        x: 1D array

    Returns:
        phase: 1D array (wrapped to [-pi, pi])
    """
    x = np.asarray(x, dtype=float)
    z = hilbert(x)
    return np.angle(z)

def hilbert_phase(x: np.ndarray) -> np.ndarray:
    """
    Backwards-compatible alias for analytic_signal_phase.
    """
    return analytic_signal_phase(x)

def unwrap_phase(phi: np.ndarray) -> np.ndarray:
    """
    Unwrap a phase time-series.
    """
    phi = np.asarray(phi, dtype=float)
    return np.unwrap(phi)


def phase_diff(phi_i: np.ndarray, phi_j: np.ndarray, wrap: bool = True) -> np.ndarray:
    """
    Phase difference Δφ = φ_j - φ_i, optionally wrapped to [-pi, pi].
    """
    d = np.asarray(phi_j, dtype=float) - np.asarray(phi_i, dtype=float)
    if wrap:
        return (d + np.pi) % (2.0 * np.pi) - np.pi
    return d


def bandpass_then_phase(
    x: np.ndarray,
    fs: float,
    f_lo: float,
    f_hi: float,
    order: int = 4,
) -> np.ndarray:
    """
    Convenience wrapper: bandpass filter then extract phase via Hilbert.
    (Filtering is intentionally omitted here to keep dependencies minimal in the core;
    implement your preferred bandpass in your application layer if needed.)
    """
    _ = (fs, f_lo, f_hi, order)
    return analytic_signal_phase(x)


def compute_phases_multichannel(
    X: np.ndarray,
    method: str = "hilbert",
    unwrap: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Compute phases for multichannel data.

    Args:
        X: array (T, N)
        method: currently 'hilbert'
        unwrap: if True, unwrap phases

    Returns:
        phases: (T, N)
        amp: optional amplitude proxy (T, N), returned for Hilbert method
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (T, N)")

    if method != "hilbert":
        raise ValueError(f"Unsupported method: {method}")

    z = hilbert(X, axis=0)
    phases = np.angle(z)
    amp = np.abs(z)

    if unwrap:
        phases = np.unwrap(phases, axis=0)

    return phases, amp
