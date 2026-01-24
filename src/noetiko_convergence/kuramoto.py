--- a/src/noetiko_convergence/kuramoto.py
+++ b/src/noetiko_convergence/kuramoto.py
@@ -1,6 +1,7 @@
 from __future__ import annotations
 
 from dataclasses import dataclass
+from typing import Optional
 
 import numpy as np


@dataclass(frozen=True)
class KuramotoSimResult:
    t: np.ndarray
    theta: np.ndarray  # shape (n_osc, n_steps)


def simulate_kuramoto_em(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    n_steps: int,
    rng: Optional[np.random.Generator] = None,
    theta0: Optional[np.ndarray] = None,
) -> KuramotoSimResult:
    """Euler–Maruyama simulation of globally-coupled noisy Kuramoto model.

    dθ_i = [ ω_i + (K/N) Σ_j sin(θ_j-θ_i) ] dt + sqrt(2D) dW_i

    Parameters
    ----------
    omega : np.ndarray
        Natural frequencies (n_osc,).
    K : float
        Effective coupling.
    D : float
        Phase noise intensity.
    dt : float
        Time step.
    n_steps : int
        Number of steps.
    rng : np.random.Generator, optional
    theta0 : np.ndarray, optional
        Initial phases (n_osc,). If None, uniform random in [0,2π).

    Returns
    -------
    KuramotoSimResult
    """
    rng = rng or np.random.default_rng()
    omega = np.asarray(omega, dtype=float)
    n = omega.size
    if n < 2:
        raise ValueError("Need at least 2 oscillators.")
    if dt <= 0 or n_steps < 2:
        raise ValueError("Invalid dt or n_steps.")

    if theta0 is None:
        theta = rng.uniform(0, 2*np.pi, size=n)
    else:
        theta = np.asarray(theta0, dtype=float).copy()
        if theta.shape != (n,):
            raise ValueError("theta0 must have shape (n_osc,).")

    out = np.empty((n, n_steps), dtype=float)
    out[:, 0] = theta
    sqrt_2Ddt = np.sqrt(max(0.0, 2.0*D*dt))

    for k in range(1, n_steps):
        # coupling term
        diff = theta[None, :] - theta[:, None]  # θ_j - θ_i
        coupling = (K / n) * np.sum(np.sin(diff), axis=1)
        drift = omega + coupling
        noise = sqrt_2Ddt * rng.standard_normal(size=n)
        theta = theta + drift*dt + noise
        # keep within [-pi,pi] for numerical stability (optional)
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        out[:, k] = theta

    t = np.arange(n_steps) * dt
    return KuramotoSimResult(t=t, theta=out)
