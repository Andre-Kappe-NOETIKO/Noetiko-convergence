#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Protocol / Metrics
# ----------------------------

@dataclass(frozen=True)
class Protocol:
    N: int = 50
    dt: float = 0.01
    D: float = 0.02
    steps: int = 5000
    M: int = 50  # ensemble size per K
    K_sweep: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0)
    w_mean: float = 0.0
    w_std: float = 0.5
    R_threshold: float = 0.5
    window: int = 200
    seed: int = 1

@dataclass(frozen=True)
class KStats:
    K: float
    r_mean: float
    r_q10: float
    r_q90: float
    tsync_median: float | None
    tsync_q10: float | None
    tsync_q90: float | None
    success_frac: float


def order_parameter_series(theta_traj: np.ndarray) -> np.ndarray:
    """
    theta_traj: (T, N) array
    returns r(t): (T,) array
    """
    z = np.mean(np.exp(1j * theta_traj), axis=1)
    return np.abs(z)


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    if window > x.size:
        return np.array([], dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="valid")


def time_to_sync(r: np.ndarray, dt: float, threshold: float, window: int) -> float | None:
    rs = rolling_mean(r, window)
    if rs.size == 0:
        return None
    idx = np.where(rs > threshold)[0]
    if idx.size == 0:
        return None
    # Convert index to time; adjust by half-window to reduce phase lag bias
    t_idx = int(idx[0] + window // 2)
    return float(t_idx * dt)


# ----------------------------
# Kuramoto engine (self-contained)
# Euler–Maruyama, global coupling
# ----------------------------

def simulate_kuramoto_em_core(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    theta0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    dθ_i = ω_i dt + K r sin(ψ - θ_i) dt + sqrt(2D) dW_i
    returns theta trajectory: shape (steps, N)
    """
    omega = np.asarray(omega, dtype=float)
    theta = np.asarray(theta0, dtype=float).copy()
    n = omega.size

    out = np.empty((steps, n), dtype=float)
    noise_scale = math.sqrt(max(0.0, 2.0 * D * dt))

    for t in range(steps):
        Z = np.mean(np.exp(1j * theta))
        r = float(np.abs(Z))
        psi = float(np.angle(Z))
        theta += omega * dt + (K * r * np.sin(psi - theta) * dt)
        theta += noise_scale * rng.standard_normal(size=n)
        out[t] = theta

    return out


# ----------------------------
# Experiment
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_ks(values: str) -> tuple[float, ...]:
    # "0,0.1,0.2" -> tuple
    parts = [p.strip() for p in values.split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def save_versions(outdir: str) -> None:
    import numpy  # noqa
    import matplotlib  # noqa

    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(outdir, "versions.json"), "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)


def save_protocol(outdir: str, protocol: Protocol) -> None:
    p = asdict(protocol)
    p["K_sweep"] = list(protocol.K_sweep)
    with open(os.path.join(outdir, "protocol.json"), "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2)


def compute_stats(vals: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(np.percentile(arr, 10)), float(np.percentile(arr, 90))


def compute_quantiles(vals: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 10)), float(np.percentile(arr, 90))


def main() -> int:
    ap = argparse.ArgumentParser(description="Kuramoto sweep: metrics + publication plots.")
    ap.add_argument("--outdir", default="results/kuramoto", help="Output directory")
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--D", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--K", type=str, default="0.0,0.1,0.2,0.4,0.8,1.2,1.6,2.0,2.5,3.0")
    ap.add_argument("--w-mean", type=float, default=0.0)
    ap.add_argument("--w-std", type=float, default=0.5)
    ap.add_argument("--R-threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    K_sweep = parse_ks(args.K)
    protocol = Protocol(
        N=args.N,
        dt=args.dt,
        D=args.D,
        steps=args.steps,
        M=args.M,
        K_sweep=K_sweep,
        w_mean=args.w_mean,
        w_std=args.w_std,
        R_threshold=args.R_threshold,
        window=args.window,
        seed=args.seed,
    )

    ensure_dir(args.outdir)
    save_protocol(args.outdir, protocol)
    save_versions(args.outdir)

    # Storage
    r_bars: dict[float, list[float]] = {K: [] for K in protocol.K_sweep}
    t_syncs: dict[float, list[float]] = {K: [] for K in protocol.K_sweep}

    base_rng = np.random.default_rng(protocol.seed)

    # Paired design per trial m: same omega + theta0 across all K
    for m in range(protocol.M):
        trial_seed = int(base_rng.integers(0, 2**31 - 1))
        rng_trial = np.random.default_rng(trial_seed)

        omega = rng_trial.normal(protocol.w_mean, protocol.w_std, size=protocol.N)
        theta0 = rng_trial.uniform(0.0, 2.0 * np.pi, size=protocol.N)

        for K in protocol.K_sweep:
            # independent, reproducible noise stream per (m,K)
            k_seed = int(rng_trial.integers(0, 2**31 - 1))
            rng_k = np.random.default_rng(k_seed)

            theta_traj = simulate_kuramoto_em_core(
                omega=omega,
                K=K,
                D=protocol.D,
                dt=protocol.dt,
                steps=protocol.steps,
                theta0=theta0,
                rng=rng_k,
            )
            r = order_parameter_series(theta_traj)

            # steady-state R: last 25%
            cutoff = int(0.75 * r.size)
            r_bar = float(np.mean(r[cutoff:]))
            r_bars[K].append(r_bar)

            ts = time_to_sync(r, protocol.dt, protocol.R_threshold, protocol.window)
            if ts is not None:
                t_syncs[K].append(float(ts))

    # Aggregate stats per K
    rows: list[KStats] = []
    for K in protocol.K_sweep:
        r_mean, r_q10, r_q90 = compute_stats(r_bars[K])
        ts_vals = t_syncs[K]
        if len(ts_vals) > 0:
            ts_med, ts_q10, ts_q90 = compute_quantiles(ts_vals)
            success_frac = float(len(ts_vals) / protocol.M)
            rows.append(KStats(K, r_mean, r_q10, r_q90, ts_med, ts_q10, ts_q90, success_frac))
        else:
            rows.append(KStats(K, r_mean, r_q10, r_q90, None, None, None, 0.0))

    # Write CSV
    csv_path = os.path.join(args.outdir, "kuramoto_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["K", "R_mean", "R_q10", "R_q90", "Tsync_median_s", "Tsync_q10_s", "Tsync_q90_s", "success_frac"]
        )
        for r in rows:
            w.writerow(
                [
                    r.K,
                    r.r_mean,
                    r.r_q10,
                    r.r_q90,
                    r.tsync_median if r.tsync_median is not None else "",
                    r.tsync_q10 if r.tsync_q10 is not None else "",
                    r.tsync_q90 if r.tsync_q90 is not None else "",
                    r.success_frac,
                ]
            )

    # Plots (publication-clean, no “chartjunk”)
    ks = np.array([r.K for r in rows], dtype=float)
    r_mean = np.array([r.r_mean for r in rows], dtype=float)
    r_q10 = np.array([r.r_q10 for r in rows], dtype=float)
    r_q90 = np.array([r.r_q90 for r in rows], dtype=float)

    ts_med = np.array([np.nan if r.tsync_median is None else r.tsync_median for r in rows], dtype=float)
    ts_q10 = np.array([np.nan if r.tsync_q10 is None else r.tsync_q10 for r in rows], dtype=float)
    ts_q90 = np.array([np.nan if r.tsync_q90 is None else r.tsync_q90 for r in rows], dtype=float)

    # Figure 1: R vs K
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    ax1.plot(ks, r_mean, lw=2)
    ax1.fill_between(ks, r_q10, r_q90, alpha=0.2)
    ax1.set_title("Steady State Synchronization", fontsize=14)
    ax1.set_xlabel("Coupling K")
    ax1.set_ylabel("Order Parameter R")
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    fig1_path = os.path.join(args.outdir, "kuramoto_R_vs_K.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # Figure 2: Tsync vs K
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=300)
    mask = ~np.isnan(ts_med)
    ax2.plot(ks[mask], ts_med[mask], marker="o", lw=2)
    ax2.fill_between(ks[mask], ts_q10[mask], ts_q90[mask], alpha=0.2)
    ax2.set_title("Time to Synchronization", fontsize=14)
    ax2.set_xlabel("Coupling K")
    ax2.set_ylabel("Time (s)")
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    fig2_path = os.path.join(args.outdir, "kuramoto_Tsync_vs_K.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {fig1_path}")
    print(f"Saved: {fig2_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
