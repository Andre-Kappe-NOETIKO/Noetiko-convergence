from __future__ import annotations

from typing import Optional

import numpy as np


def simulate_kuramoto_em(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    theta0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    *,
    n_steps: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate noisy globally-coupled Kuramoto phases via Euler–Maruyama.

    dθ_i = ω_i dt + (K/N) Σ_j sin(θ_j - θ_i) dt + sqrt(2D) dW_i

    Compatibility:
      - 'steps' is the historical parameter (positional / keyword).
      - 'n_steps' is an alias required by tests.
      - If both are provided and differ -> ValueError.
      - If rng is provided, it is used; otherwise default_rng(seed).

    Returns:
      theta: array of shape (num_steps, N)  (NOTE: no initial state row)
    """
    # --- Resolve steps / n_steps ---
    if n_steps is not None:
        if steps is not None and steps != n_steps:
            raise ValueError("'steps' and 'n_steps' must match if both are provided.")
        steps = n_steps

    if steps is None:
        raise ValueError("Must provide 'steps' or 'n_steps'.")

    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("'steps' must be a positive integer.")

    if dt <= 0:
        raise ValueError("'dt' must be positive.")

    omega = np.asarray(omega, dtype=float)
    n = int(omega.size)
    if n == 0:
        raise ValueError("'omega' must be a non-empty array.")

    # --- RNG precedence: rng overrides seed ---
    if rng is None:
        rng = np.random.default_rng(seed)

    # --- Initial conditions ---
    if theta0 is None:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n).astype(float, copy=False)
    else:
        theta0 = np.asarray(theta0, dtype=float)
        if theta0.shape != (n,):
            raise ValueError(f"'theta0' must have shape ({n},), got {theta0.shape}.")
        theta = theta0.copy()

    out = np.empty((steps, n), dtype=float)

    # Clamp D to avoid sqrt of negative due to accidental sign
    D_eff = float(D) if D > 0.0 else 0.0
    sqrt_2d_dt = np.sqrt(2.0 * D_eff * dt)

    # --- EM integration ---
    for t in range(steps):
        # Efficient mean-field coupling:
        # K*r*sin(psi - theta) == K*(<sinθ>*cosθ - <cosθ>*sinθ)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        mean_sin = float(np.mean(sin_theta))
        mean_cos = float(np.mean(cos_theta))

        drift = omega + K * (mean_sin * cos_theta - mean_cos * sin_theta)
        noise = sqrt_2d_dt * rng.standard_normal(size=n)

        theta = theta + dt * drift + noise
        out[t] = theta

    return out
