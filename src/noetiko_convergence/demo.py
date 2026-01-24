from __future__ import annotations

import numpy as np

from .kuramoto import order_parameter, simulate_kuramoto_global
from .phase import compute_phases_multichannel
from .quality_gates import (
    consistency_gate,
    robustness_gate,
    spectral_gate,
)
from .surrogates import phase_randomization_multichannel, time_shift_surrogate


def run_demo(seed: int = 0) -> dict[str, float]:
    """
    Proof-of-protocol demo:
    - simulate multichannel oscillatory data (as a stand-in)
    - extract phases
    - apply quality gates
    - compute r(t)
    - compare against mandatory surrogates

    Returns:
        summary dict (non-interpretive).
    """
    rng = np.random.default_rng(seed)

    # synthetic phases via Kuramoto (stand-in for any multichannel dataset)
    N = 12
    omega = rng.normal(loc=0.0, scale=0.8, size=N)
    theta = simulate_kuramoto_global(omega, K=1.7, D=0.08, dt=0.02, steps=2500, seed=seed)

    # create synthetic observables x_i(t) from phases (cosine projection + noise)
    X = np.cos(theta) + 0.12 * rng.standard_normal(size=theta.shape)

    # extract phases
    phases, amp = compute_phases_multichannel(X, method="hilbert", unwrap=True)

    # gates (non-exhaustive minimal set)
    g1 = spectral_gate(X, fs=50.0, f_lo=0.5, f_hi=10.0)
    g2 = robustness_gate(phases, amp=amp)
    g3 = consistency_gate(phases)

    passed = float(g1 and g2 and g3)

    r_t = order_parameter(phases)

    # surrogates
    X_pr = phase_randomization_multichannel(X, rng=rng)
    phases_pr, _ = compute_phases_multichannel(X_pr, method="hilbert", unwrap=True)
    r_pr = order_parameter(phases_pr)

    X_ts = time_shift_surrogate(X, shift=250)
    phases_ts, _ = compute_phases_multichannel(X_ts, method="hilbert", unwrap=True)
    r_ts = order_parameter(phases_ts)

    # simple separation metric
    sep_pr = float(np.mean(r_t) - np.mean(r_pr))
    sep_ts = float(np.mean(r_t) - np.mean(r_ts))

    return {
        "passed_gates": passed,
        "mean_r": float(np.mean(r_t)),
        "mean_r_phase_rand": float(np.mean(r_pr)),
        "mean_r_time_shift": float(np.mean(r_ts)),
        "sep_phase_rand": sep_pr,
        "sep_time_shift": sep_ts,
    }
