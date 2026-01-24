from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KuramotoParams:
    """Parameter container for the noisy globally-coupled Kuramoto model."""

    K: float
    D: float
    dt: float
    steps: int
    seed: Optional[int] = None


def simulate_kuramoto_em(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    theta0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Euler–Maruyama simulation for noisy globally-coupled Kuramoto phases.

    SDE:
        dθ_i = ω_i dt + (K/N) Σ_j sin(θ_j - θ_i) dt + sqrt(2D) dW_i

    Using the identity:
        (K/N) Σ_j sin(θ_j - θ_i) = K r sin(ψ - θ_i)
        where r e^{iψ} = (1/N) Σ_j e^{iθ_j}

    Args:
        omega: shape (N,), natural frequencies.
        K: coupling strength.
        D: phase noise intensity (>= 0).
        dt: time step (> 0).
        steps: number of steps (> 0).
        theta0: optional initial phases, shape (N,).
        seed: RNG seed.

    Returns:
        theta: array of shape (steps, N) with phases.
    """
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1:
        raise ValueError("omega must be a 1D array of shape (N,)")

    if dt <= 0:
        raise ValueError("dt must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if D < 0:
        raise ValueError("D must be >= 0")

    n = omega.size
    rng = np.random.default_rng(seed)

    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(float)
    else:
        theta = np.asarray(theta0, dtype=float).copy()
        if theta.shape != (n,):
            raise ValueError(f"theta0 must have shape {(n,)}, got {theta.shape}")

    out = np.empty((steps, n), dtype=float)
    noise_scale = np.sqrt(2.0 * D * dt) if D > 0 else 0.0

    for t in range(steps):
        z = np.mean(np.exp(1j * theta))  # complex order parameter
        r = np.abs(z)
        psi = np.angle(z)

        drift = omega + K * r * np.sin(psi - theta)
        theta = theta + drift * dt

        if noise_scale > 0:
            theta = theta + noise_scale * rng.standard_normal(size=n)

        out[t] = theta

    return out


def simulate_kuramoto_global(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    theta0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Backwards-compatible alias."""
    return simulate_kuramoto_em(omega=omega, K=K, D=D, dt=dt, steps=steps, theta0=theta0, seed=seed)


def order_parameter(theta: np.ndarray) -> np.ndarray:
    """
    Compute r(t) from phases.

    Args:
        theta: shape (T, N) or (N,).

    Returns:
        r: shape (T,) if theta is (T, N), else shape (1,).
    """
    theta = np.asarray(theta, dtype=float)

    if theta.ndim == 1:
        return np.array([np.abs(np.mean(np.exp(1j * theta)))], dtype=float)

    if theta.ndim != 2:
        raise ValueError("theta must have shape (N,) or (T, N)")

    return np.abs(np.mean(np.exp(1j * theta), axis=1))
