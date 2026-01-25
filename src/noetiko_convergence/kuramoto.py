from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KuramotoSimResult:
    """Simulation result container.

    Attributes
    ----------
    t : np.ndarray
        Time grid, shape (n_steps,).
    theta : np.ndarray
        Phase trajectory, shape (n_osc, n_steps).
        Time is along axis=1 to match tests and common signal conventions.
    """

    t: np.ndarray
    theta: np.ndarray  # (n_osc, n_steps)


def simulate_kuramoto_em(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    n_steps: Optional[int] = None,
    *,
    steps: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    theta0: Optional[np.ndarray] = None,
    wrap: bool = True,
) -> KuramotoSimResult:
    """Euler–Maruyama simulation of the globally-coupled noisy Kuramoto model.

    Model:
        dθ_i = [ ω_i + (K/N) Σ_j sin(θ_j - θ_i) ] dt + sqrt(2D) dW_i

    Notes
    -----
    - API is aligned with tests: accepts `n_steps` and `rng`.
    - Backward-compatible alias: `steps` (keyword-only).
    - RNG precedence: if `rng` is given, it is used; otherwise `seed` is used.

    Parameters
    ----------
    omega : np.ndarray
        Natural frequencies, shape (n_osc,).
    K : float
        Coupling strength.
    D : float
        Phase noise intensity.
    dt : float
        Time step (> 0).
    n_steps : int, optional
        Number of steps (preferred name; used by tests).
    steps : int, optional
        Alias for n_steps (kept for backward compatibility).
    rng : np.random.Generator, optional
        Random generator (preferred over seed).
    seed : int, optional
        Seed used only if rng is None.
    theta0 : np.ndarray, optional
        Initial phases, shape (n_osc,). If None: uniform in [0, 2π).
    wrap : bool
        If True, wrap phases into [-π, π] each step.

    Returns
    -------
    KuramotoSimResult
        t shape (n_steps,), theta shape (n_osc, n_steps).
    """
    # ---- resolve steps / n_steps ----
    if n_steps is None and steps is None:
        raise ValueError("Provide `n_steps` (preferred) or `steps` (alias).")
    if n_steps is not None and steps is not None and n_steps != steps:
        raise ValueError("If both provided, `n_steps` and `steps` must match.")
    n_steps_final = int(n_steps if n_steps is not None else steps)  # type: ignore[arg-type]

    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if n_steps_final < 2:
        raise ValueError("n_steps must be >= 2.")
    if D < 0:
        raise ValueError("D must be >= 0.")

    # ---- rng ----
    if rng is None:
        rng = np.random.default_rng(seed)

    # ---- inputs ----
    omega = np.asarray(omega, dtype=float)
    n = int(omega.size)
    if n < 2:
        raise ValueError("Need at least 2 oscillators (len(omega) >= 2).")

    # ---- init theta ----
    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    else:
        theta = np.asarray(theta0, dtype=float).copy()
        if theta.shape != (n,):
            raise ValueError(f"theta0 must have shape ({n},).")

    # ---- allocate output: match tests (n_osc, n_steps) ----
    out = np.empty((n, n_steps_final), dtype=float)
    out[:, 0] = theta
    t = np.arange(n_steps_final, dtype=float) * dt

    sqrt_2Ddt = np.sqrt(2.0 * D * dt) if D > 0 else 0.0

    # ---- EM integration ----
    for k in range(1, n_steps_final):
        # mean-field identity:
        # (K/N) Σ_j sin(θ_j - θ_i) = K * r * sin(ψ - θ_i)
        order = np.mean(np.exp(1j * theta))
        r = np.abs(order)
        psi = np.angle(order)

        drift = omega + K * r * np.sin(psi - theta)
        noise = sqrt_2Ddt * rng.standard_normal(size=n) if sqrt_2Ddt > 0 else 0.0

        theta = theta + drift * dt + noise

        if wrap:
            theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

        out[:, k] = theta

    return KuramotoSimResult(t=t, theta=out)
