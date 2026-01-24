from __future__ import annotations

from typing import Optional

import numpy as np


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
    Euler–Maruyama simulation for noisy globally-coupled Kuramoto phases.

    SDE:
        dθ_i = ω_i dt + (K/N) Σ_j sin(θ_j - θ_i) dt + sqrt(2D) dW_i

    API notes (for test-compatibility):
      - accepts `n_steps` as an alias for `steps`
      - accepts `rng` (np.random.Generator). If provided, it overrides `seed`.

    Args:
        omega: shape (N,), natural frequencies.
        K: coupling strength.
        D: phase noise intensity (>= 0).
        dt: time step (> 0).
        steps: number of time steps (legacy name).
        n_steps: number of time steps (preferred in tests).
        theta0: optional initial phases, shape (N,).
        seed: RNG seed (used only if rng is None).
        rng: numpy Generator (preferred).

    Returns:
        theta: array of shape (n_steps, N) with phases.
    """
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1:
        raise ValueError("omega must be a 1D array of shape (N,)")

    if dt <= 0:
        raise ValueError("dt must be > 0")
    if D < 0:
        raise ValueError("D must be >= 0")

    # Resolve number of steps (support both names)
    if n_steps is None and steps is None:
        raise TypeError("You must provide `n_steps` or `steps`.")
    if n_steps is not None and steps is not None and n_steps != steps:
        raise ValueError("If both `n_steps` and `steps` are provided, they must match.")

    T = int(n_steps if n_steps is not None else steps)  # final step count
    if T <= 0:
        raise ValueError("Number of steps must be > 0")

    n = omega.size

    # Resolve RNG
    if rng is None:
        rng = np.random.default_rng(seed)

    # Init phases
    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(float)
    else:
        theta = np.asarray(theta0, dtype=float).copy()
        if theta.shape != (n,):
            raise ValueError(f"theta0 must have shape {(n,)}, got {theta.shape}")

    out = np.empty((T, n), dtype=float)
    noise_scale = np.sqrt(2.0 * D * dt) if D > 0 else 0.0

    for t in range(T):
        z = np.mean(np.exp(1j * theta))  # complex order parameter
        r = np.abs(z)
        psi = np.angle(z)

        drift = omega + K * r * np.sin(psi - theta)
        theta = theta + drift * dt

        if noise_scale > 0:
            theta = theta + noise_scale * rng.standard_normal(size=n)

        out[t] = theta

    return out
