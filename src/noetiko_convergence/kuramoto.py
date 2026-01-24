from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KuramotoParams:
    K: float
    D: float
    dt: float
    steps: int
    seed: Optional[int] = None


def simulate_kuramoto_global(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    theta0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate noisy globally-coupled Kuramoto phases.

    dθ_i = ω_i dt + (K/N) Σ_j sin(θ_j - θ_i) dt + sqrt(2D) dW_i

    Returns:
        theta: array of shape (steps, N)
    """
    rng = np.random.default_rng(seed)
    omega = np.asarray(omega, dtype=float)
    n = omega.size

    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    else:
        theta = np.asarray(theta0, dtype=float).copy()

    out = np.empty((steps, n), dtype=float)
    sqrt_2d_dt = np.sqrt(max(0.0, 2.0 * D * dt))

    for t in range(steps):
        order = np.mean(np.exp(1j * theta))
        psi = np.angle(order)
        r = np.abs(order)

        theta += omega * dt + K * r * np.sin(psi - theta) * dt
        theta += sqrt_2d_dt * rng.standard_normal(size=n)

        out[t] = theta

    return out


def order_parameter(theta: np.ndarray) -> np.ndarray:
    """
    Compute r(t) for theta shaped (T, N) or (N,) (broadcasted).
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        return np.array([np.abs(np.mean(np.exp(1j * theta)))], dtype=float)

    return np.abs(np.mean(np.exp(1j * theta), axis=1))
