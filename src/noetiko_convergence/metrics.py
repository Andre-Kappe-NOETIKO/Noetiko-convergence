from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class OrderParameterResult:
    r: np.ndarray
    psi: np.ndarray


def order_parameter(theta: np.ndarray, axis: int = 0) -> OrderParameterResult:
    """Compute Kuramoto order parameter r e^{i psi}.

    Parameters
    ----------
    theta : np.ndarray
        Phases in radians. Can be shaped (n_oscillators, n_samples) or similar.
    axis : int
        Axis over which to average oscillators (default 0).

    Returns
    -------
    OrderParameterResult with r and psi arrays (shape = theta with axis removed).
    """
    theta = np.asarray(theta, dtype=float)
    z = np.exp(1j * theta)
    m = np.mean(z, axis=axis)
    r = np.abs(m)
    psi = np.angle(m)
    return OrderParameterResult(r=r, psi=psi)


def metastability_index(r_t: np.ndarray) -> float:
    """Metastability index as variance of r(t) across time."""
    r_t = np.asarray(r_t, dtype=float)
    return float(np.var(r_t))


def phase_locking_value(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """Phase-locking value between two phase time series."""
    theta_a = np.asarray(theta_a, dtype=float)
    theta_b = np.asarray(theta_b, dtype=float)
    d = theta_a - theta_b
    return float(np.abs(np.mean(np.exp(1j * d))))
