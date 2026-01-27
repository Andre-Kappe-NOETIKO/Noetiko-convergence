from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KuramotoSimResult:
    """Container for Kuramoto simulation output.

    Parameters
    ----------
    t:
        Time grid, shape (n_steps,).
    theta:
        Phase trajectory, shape (n_osc, n_steps). Time runs along axis=1.
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
    """Euler–Maruyama simulation of the globally coupled noisy Kuramoto model.

    Model
    -----
        dθ_i = [ ω_i + (K/N) Σ_j sin(θ_j - θ_i) ] dt + sqrt(2D) dW_i

    API notes
    ---------
    - `n_steps` is the preferred argument.
    - `steps` is a backward-compatible alias (keyword-only).
    - RNG precedence: if `rng` is provided, it is used; otherwise `seed` is used.

    Parameters
    ----------
    omega:
        Natural frequencies, shape (n_osc,).
    K:
        Coupling strength.
    D:
        Phase noise intensity (>= 0).
    dt:
        Time step (> 0).
    n_steps:
        Number of steps (>= 2).
    steps:
        Alias for `n_steps` (must match if both are given).
    rng:
        NumPy random generator.
    seed:
        Seed used only if `rng` is None.
    theta0:
        Initial phases, shape (n_osc,). If None: uniform in [0, 2π).
    wrap:
        If True, wrap phases to [-π, π] after each step.

    Returns
    -------
    KuramotoSimResult
        t shape (n_steps,), theta shape (n_osc, n_steps).
    """
    # -------- resolve n_steps / steps --------
    if n_steps is None and steps is None:
        raise ValueError("Provide `n_steps` (preferred) or `steps` (alias).")
    if n_steps is not None and steps is not None and int(n_steps) != int(steps):
        raise ValueError("If both provided, `n_steps` and `steps` must match.")
    n_steps_final = int(n_steps if n_steps is not None else steps)  # type: ignore[arg-type]

    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if n_steps_final < 2:
        raise ValueError("n_steps must be >= 2.")
    if D < 0:
        raise ValueError("D must be >= 0.")

    # -------- RNG --------
    if rng is None:
        rng = np.random.default_rng(seed)

    # -------- inputs --------
    omega_arr = np.asarray(omega, dtype=float).reshape(-1)
    n = int(omega_arr.size)
    if n < 2:
        raise ValueError("Need at least 2 oscillators (len(omega) >= 2).")

    # -------- init theta --------
    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(float)
    else:
        theta = np.asarray(theta0, dtype=float).reshape(-1).copy()
        if theta.shape != (n,):
            raise ValueError(f"theta0 must have shape ({n},).")

    # -------- allocate output (n_osc, n_steps) --------
    out = np.empty((n, n_steps_final), dtype=float)
    out[:, 0] = theta
    t = np.arange(n_steps_final, dtype=float) * float(dt)

    sqrt_2Ddt = float(np.sqrt(2.0 * D * dt)) if D > 0 else 0.0

    # -------- integration --------
    for k in range(1, n_steps_final):
        # Mean-field identity:
        # (K/N) Σ_j sin(θ_j - θ_i) = K * r * sin(ψ - θ_i)
        z = np.mean(np.exp(1j * theta))
        r = float(np.abs(z))
        psi = float(np.angle(z))

        drift = omega_arr + K * r * np.sin(psi - theta)

        if sqrt_2Ddt > 0.0:
            theta = theta + drift * dt + sqrt_2Ddt * rng.standard_normal(size=n)
        else:
            theta = theta + drift * dt

        if wrap:
            theta = (theta + np.pi) % (2.0 * np.pi) - np.pi

        out[:, k] = theta

    return KuramotoSimResult(t=t, theta=out)


def order_parameter(theta: np.ndarray, axis: int = -1) -> np.ndarray:
    """Kuramoto order parameter r = |mean(exp(i*theta))|.

    Parameters
    ----------
    theta:
        Phase array, e.g. shape (n_osc,) or (n_osc, n_steps) etc.
    axis:
        Axis over which to average (default: last axis).

    Returns
    -------
    np.ndarray
        Order parameter magnitude. Scalar if the mean collapses all axes.
    """
    th = np.asarray(theta, dtype=float)
    z = np.mean(np.exp(1j * th), axis=axis)
    return np.abs(z)


# Backward-compatible alias (if older code expects this name)
simulate_kuramoto_global = simulate_kuramoto_em
