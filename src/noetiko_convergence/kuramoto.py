import numpy as np
from typing import Optional

def simulate_kuramoto_em(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: Optional[int] = None,
    theta0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    n_steps: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate the Kuramoto model with noise using Euler-Maruyama method.
    
    Parameters:
    - omega: Array of natural frequencies (shape N)
    - K: Coupling strength
    - D: Noise intensity
    - dt: Time step
    - steps: Number of simulation steps (backward compatible)
    - theta0: Initial phases (optional)
    - seed: Random seed (if rng not provided)
    - n_steps: Alias for steps (for test compatibility)
    - rng: Random number generator (optional)
    
    Returns:
    - Trajectory of phases: shape (steps + 1, N)
    """
    if steps is None and n_steps is None:
        raise ValueError("Must provide either 'steps' or 'n_steps'")
    if steps is not None and n_steps is not None and steps != n_steps:
        raise ValueError("'steps' and 'n_steps' must be equal if both provided")
    
    num_steps = steps if steps is not None else n_steps
    
    if num_steps is None or num_steps <= 0:
        raise ValueError("Number of steps must be a positive integer")
    
    N = len(omega)
    
    if rng is None:
        rng = np.random.default_rng(seed)
    
    if theta0 is None:
        theta = rng.uniform(-np.pi, np.pi, N)
    else:
        if theta0.shape != (N,):
            raise ValueError(f"theta0 must have shape ({N},)")
        theta = theta0.copy()
    
    trajectory = np.zeros((num_steps + 1, N))
    trajectory[0] = theta
    
    sqrt_2D_dt = np.sqrt(2 * D * dt)
    
    for t in range(1, num_steps + 1):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        mean_sin = np.mean(sin_theta)
        mean_cos = np.mean(cos_theta)
        
        drift = omega + K * (mean_sin * cos_theta - mean_cos * sin_theta)
        diffusion = sqrt_2D_dt * rng.standard_normal(N)
        
        theta += dt * drift + diffusion
        # Optional: theta %= 2 * np.pi  # But not necessary for most analyses
        
        trajectory[t] = theta
    
    return trajectory
