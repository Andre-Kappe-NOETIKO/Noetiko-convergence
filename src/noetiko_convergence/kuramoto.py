from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KuramotoParams:
    """Parameters for a noisy globally-coupled Kuramoto simulation (Euler–Maruyama)."""

    K: float
    D: float
    dt: float
    steps: int
    seed: Optional[int] = None


def _validate_inputs(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    theta0: Optional[np.ndarray],
) -> tuple[np.ndarray, float, float, float, int, Optional[np.ndarray]]:
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1 or omega.size == 0:
        raise ValueError("omega must be a non-empty 1D array of shape (N,)")

    if not np.isfinite(K):
        raise ValueError("K must be finite")
    if not np.isfinite(D) or D < 0.0:
        raise ValueError("D must be finite and >= 0")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and > 0")
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be an int > 0")

    if theta0 is not None:
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.shape != omega.shape:
            raise ValueError("theta0 must have the same shape as omega, i.e., (N,)")

    return omega, float(K), float(D), float(dt), int(steps), theta0


def simulate_kuramoto_global(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    theta0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    r"""
    Simulate noisy globally-coupled Kuramoto phases via Euler–Maruyama.

    SDE (Ito):
        dθ_i = ω_i dt + (K/N) Σ_j sin(θ_j - θ_i) dt + sqrt(2D) dW_i

    Efficient mean-field form using the complex order parameter:
        re^{iψ} = (1/N) Σ_j e^{iθ_j}
        (K/N) Σ_j sin(θ_j - θ_i) = K r sin(ψ - θ_i)

    Args:
        omega: Natural frequencies, shape (N,).
        K: Coupling strength (effective).
        D: Phase-noise intensity (>=0).
        dt: Time step (>0).
        steps: Number of integration steps (>0).
        theta0: Optional initial phases, shape (N,). If None, uniform on [0, 2π).
        seed: Optional RNG seed for reproducibility.

    Returns:
        theta: Phase trajectory of shape (steps, N).

    Notes:
        - This routine does not wrap phases to [-π, π]. That is intentional for continuity.
          If you need wrapped phases, apply: (theta + π) % (2π) - π.
        - For deterministic runs set D=0 and/or provide a fixed seed.
    """
    omega, K, D, dt, steps, theta0 = _validate_inputs(omega, K, D, dt, steps, theta0)
    rng = np.random.default_rng(seed)

    n = omega.size
    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    else:
        theta = theta0.copy()

    out = np.empty((steps, n), dtype=float)
    noise_scale = np.sqrt(2.0 * D * dt) if D > 0.0 else 0.0

    for t in range(steps):
        # complex order parameter: re^{iψ}
        z = np.mean(np.exp(1j * theta))
        r = np.abs(z)
        psi = np.angle(z)

        # drift + coupling (mean-field)
        theta = theta + omega * dt + (K * r * np.sin(psi - theta)) * dt

        # diffusion
        if noise_scale > 0.0:
            theta = theta + noise_scale * rng.standard_normal(size=n)

        out[t] = theta

    return out


def simulate_kuramoto_em(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    theta0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Backwards-compatible alias for Euler–Maruyama Kuramoto simulation."""
    return simulate_kuramoto_global(
        omega=omega,
        K=K,
        D=D,
        dt=dt,
        steps=steps,
        theta0=theta0,
        seed=seed,
    )


def order_parameter(theta: np.ndarray) -> np.ndarray:
    """
    Compute r(t) = |(1/N) Σ_j e^{iθ_j}|.

    Args:
        theta: shape (T, N) or (N,).

    Returns:
        If theta is (T, N): array shape (T,)
        If theta is (N,): array shape (1,) for consistency with tests/pipelines.
    """
    theta = np.asarray(theta, dtype=float)

    if theta.ndim == 1:
        z = np.mean(np.exp(1j * theta))
        return np.array([np.abs(z)], dtype=float)

    if theta.ndim != 2:
        raise ValueError("theta must have shape (N,) or (T, N)")

    z = np.mean(np.exp(1j * theta), axis=1)
    return np.abs(z)


def order_parameter_complex(theta: np.ndarray) -> np.ndarray:
    """
    Compute complex order parameter z(t) = (1/N) Σ_j e^{iθ_j}.

    Args:
        theta: shape (T, N) or (N,).

    Returns:
        If theta is (T, N): complex array shape (T,)
        If theta is (N,): complex array shape (1,)
    """
    theta = np.asarray(theta, dtype=float)

    if theta.ndim == 1:
        return np.array([np.mean(np.exp(1j * theta))], dtype=complex)

    if theta.ndim != 2:
        raise ValueError("theta must have shape (N,) or (T, N)")

    return np.mean(np.exp(1j * theta), axis=1)
