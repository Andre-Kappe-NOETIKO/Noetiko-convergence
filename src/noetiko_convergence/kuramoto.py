from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class KuramotoSimResult:
    """Container for a Kuramoto simulation result.

    Attributes
    ----------
    t:
        Time grid, shape (n_steps,).
    theta:
        Phase trajectories, shape (n_osc, n_steps).
        Time runs along axis=1 (matches common signal conventions).
    """

    t: np.ndarray
    theta: np.ndarray


def order_parameter(theta: np.ndarray, *, axis: int = 0) -> np.ndarray:
    """Compute the Kuramoto complex order parameter.

    Parameters
    ----------
    theta:
        Phases. Supported shapes:
        - (n_osc,) -> returns scalar complex (0-d array)
        - (n_osc, n_steps) -> returns (n_steps,) complex if axis=0
        - (n_steps, n_osc) -> returns (n_steps,) complex if axis=1
    axis:
        Axis along which oscillators are arranged.

    Returns
    -------
    np.ndarray
        Complex order parameter R = (1/N) * sum(exp(1j * theta)).
        Magnitude |R| is the order parameter r, angle is mean phase psi.
    """
    th = np.asarray(theta, dtype=float)
    return np.mean(np.exp(1j * th), axis=axis)


def _wrap_pm_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def simulate_kuramoto_global(
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
    """Euler–Maruyama simulation of globally-coupled noisy Kuramoto model.

    Model:
        dθ_i = [ ω_i + (K/N) Σ_j sin(θ_j - θ_i) ] dt + sqrt(2D) dW_i

    Notes
    -----
    - Accepts both `n_steps` (preferred) and `steps` (alias).
    - RNG precedence: if `rng` is given it is used; else `seed` is used.

    Returns
    -------
    KuramotoSimResult
        t shape (n_steps,), theta shape (n_osc, n_steps).
    """
    # --- resolve steps ---
    if n_steps is None and steps is None:
        raise ValueError("Provide `n_steps` (preferred) or `steps` (alias).")
    if n_steps is not None and steps is not None and int(n_steps) != int(steps):
        raise ValueError("If both provided, `n_steps` and `steps` must match.")
    n_steps_final = int(n_steps if n_steps is not None else steps)  # type: ignore[arg-type]

    # --- validate ---
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if n_steps_final < 2:
        raise ValueError("n_steps must be >= 2.")
    if D < 0:
        raise ValueError("D must be >= 0.")

    w = np.asarray(omega, dtype=float)
    n = int(w.size)
    if n < 2:
        raise ValueError("Need at least 2 oscillators (len(omega) >= 2).")

    # --- rng ---
    if rng is None:
        rng = np.random.default_rng(seed)

    # --- init phases ---
    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    else:
        theta = np.asarray(theta0, dtype=float).copy()
        if theta.shape != (n,):
            raise ValueError(f"theta0 must have shape ({n},).")

    # --- allocate outputs ---
    out = np.empty((n, n_steps_final), dtype=float)
    out[:, 0] = theta
    t = np.arange(n_steps_final, dtype=float) * dt

    sqrt_2Ddt = float(np.sqrt(2.0 * D * dt)) if D > 0 else 0.0

    # --- integration ---
    for k in range(1, n_steps_final):
        # mean-field identity:
        # (K/N) Σ_j sin(θ_j - θ_i) = K * r * sin(psi - θ_i)
        R = np.mean(np.exp(1j * theta))
        r = float(np.abs(R))
        psi = float(np.angle(R))

        drift = w + K * r * np.sin(psi - theta)
        if sqrt_2Ddt > 0:
            noise = sqrt_2Ddt * rng.standard_normal(size=n)
            theta = theta + drift * dt + noise
        else:
            theta = theta + drift * dt

        if wrap:
            theta = _wrap_pm_pi(theta)

        out[:, k] = theta

    return KuramotoSimResult(t=t, theta=out)


# ---- Backward-compatible alias (if other code/tests use EM naming) ----
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
    return simulate_kuramoto_global(
        omega=omega,
        K=K,
        D=D,
        dt=dt,
        n_steps=n_steps,
        steps=steps,
        rng=rng,
        seed=seed,
        theta0=theta0,
        wrap=wrap,
    )
