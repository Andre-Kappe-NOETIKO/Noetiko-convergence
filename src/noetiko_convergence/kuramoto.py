from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KuramotoParams:
    """Parameter bundle for a noisy globally-coupled Kuramoto simulation."""

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
    steps: Optional[int] = None,
    *,
    n_steps: Optional[int] = None,
    theta0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Euler–Maruyama simulation of noisy globally-coupled Kuramoto phases.

    SDE (global coupling):
        dθ_i = ω_i dt + K r sin(ψ - θ_i) dt + sqrt(2D) dW_i
    where r e^{iψ} = (1/N) Σ_j exp(i θ_j).

    This function is API-compatible with tests that call:
        simulate_kuramoto_em(..., n_steps=..., rng=...)

    Args:
        omega: Natural frequencies, shape (N,).
        K: Global coupling strength.
        D: Noise intensity (diffusion coefficient).
        dt: Time step.
        steps: Number of integration steps (legacy positional).
        n_steps: Number of integration steps (preferred keyword).
        theta0: Optional initial phases, shape (N,). If None, uniform on [0, 2π).
        seed: Seed used only if rng is None.
        rng: Optional numpy Generator. If given, takes precedence over seed.

    Returns:
        theta: Array of phases with shape (T, N), where T is steps/n_steps.
    """
    # Resolve step count from either legacy `steps` or keyword `n_steps`
    if steps is None and n_steps is None:
        raise TypeError("You must provide `steps` or `n_steps`.")
    if steps is not None and n_steps is not None and int(steps) != int(n_steps):
        raise ValueError("`steps` and `n_steps` were both provided but differ.")
    T = int(n_steps if n_steps is not None else steps)  # type: ignore[arg-type]
    if T <= 0:
        raise ValueError("Number of steps must be a positive integer.")

    # RNG handling (tests pass rng=...)
    if rng is None:
        rng = np.random.default_rng(seed)

    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1:
        raise ValueError("omega must be a 1D array of shape (N,).")
    n = omega.size
    if n == 0:
        raise ValueError("omega must contain at least one element.")

    # Initial condition
    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(float, copy=False)
    else:
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.shape != (n,):
            raise ValueError(f"theta0 must have shape ({n},), got {theta0.shape}.")
        theta = theta0.copy()

    out = np.empty((T, n), dtype=float)

    # Precompute noise scale
    # EM: theta += sqrt(2D dt) * N(0,1)
    noise_scale = float(np.sqrt(max(0.0, 2.0 * D * dt)))

    for t in range(T):
        # complex order parameter z = r * exp(i psi)
        z = np.mean(np.exp(1j * theta))
        r = float(np.abs(z))
        psi = float(np.angle(z))

        # Drift
        theta += omega * dt
        theta += K * r * np.sin(psi - theta) * dt

        # Diffusion
        if noise_scale > 0.0:
            theta += noise_scale * rng.standard_normal(size=n)

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
    """Backwards-compatible alias for older call sites."""
    return simulate_kuramoto_em(
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
    Compute r(t) from phases.

    Args:
        theta: shape (T, N) or (N,).

    Returns:
        r: shape (T,) if theta is (T, N), else shape (1,).
    """
    theta = np.asarray(theta, dtype=float)

    if theta.ndim == 1:
        z = np.mean(np.exp(1j * theta))
        return np.array([np.abs(z)], dtype=float)

    if theta.ndim != 2:
        raise ValueError("theta must have shape (N,) or (T, N).")

    z = np.mean(np.exp(1j * theta), axis=1)
    return np.abs(z)
