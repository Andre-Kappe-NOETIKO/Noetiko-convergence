#!/usr/bin/env python3
"""
Kuramoto sweep runner (Noetiko Convergence)

What it does
- Runs an ensemble sweep over coupling K for the stochastic globally-coupled Kuramoto model.
- Computes steady-state order parameter statistics and "time-to-sync" (threshold crossing) stats.
- Writes:
    - protocol.json
    - versions.json
    - kuramoto_metrics.csv
    - kuramoto_R_vs_K.png
    - kuramoto_Tsync_vs_K.png

Key design points
- Reproducible seeding strategy.
- Robust to different simulate_kuramoto_em APIs (n_steps vs steps, rng vs seed, return type).
- Uses timezone-aware UTC timestamp (fixes datetime.utcnow() deprecation warnings).
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Prefer library implementation if available
try:
    from noetiko_convergence.kuramoto import simulate_kuramoto_em  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Could not import noetiko_convergence.kuramoto.simulate_kuramoto_em. "
        "Install the package in editable mode first: python -m pip install -e ."
    ) from e


# -----------------------------
# Protocol definition
# -----------------------------
@dataclass(frozen=True)
class SweepProtocol:
    N: int = 50
    dt: float = 0.01
    D: float = 0.02
    steps: int = 5000
    M: int = 50
    w_mean: float = 0.0
    w_std: float = 0.5

    # K sweep default (matches your earlier protocol)
    K_sweep: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0)

    # Time-to-sync metric
    R_threshold: float = 0.5
    window_size: int = 200

    # Determinism
    base_seed: int = 12345


# -----------------------------
# Helpers
# -----------------------------
def _order_parameter_series(theta_TN: np.ndarray) -> np.ndarray:
    """
    theta_TN: array shaped (T, N)
    returns: r(t) shaped (T,)
    """
    z = np.mean(np.exp(1j * theta_TN), axis=1)
    return np.abs(z)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=float) / float(window)
    # valid: length T-window+1
    return np.convolve(x, kernel, mode="valid")


def _time_to_sync_seconds(r_series: np.ndarray, dt: float, threshold: float, window: int) -> float:
    """
    First time where rolling-mean(r) > threshold.
    Returns seconds, or np.nan if never crosses.
    """
    if r_series.size < max(2, window + 1):
        return float("nan")

    r_smooth = _moving_average(r_series, window)
    idx = np.where(r_smooth > threshold)[0]
    if idx.size == 0:
        return float("nan")

    # r_smooth[k] corresponds to window centered ~ k + (window-1)/2
    # We report the start-index time (conservative)
    t_sync = float(idx[0]) * dt
    return t_sync


def _call_simulator(
    omega: np.ndarray,
    K: float,
    D: float,
    dt: float,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Calls simulate_kuramoto_em robustly across possible signatures and return types.
    Returns theta time series shaped (T, N).
    """
    sig = inspect.signature(simulate_kuramoto_em)
    params = sig.parameters

    kwargs: Dict[str, Any] = {
        "omega": omega,
        "K": float(K),
        "D": float(D),
        "dt": float(dt),
    }

    # steps argument name
    if "n_steps" in params:
        kwargs["n_steps"] = int(steps)
    elif "steps" in params:
        kwargs["steps"] = int(steps)
    else:
        raise TypeError("simulate_kuramoto_em has neither 'n_steps' nor 'steps' parameter.")

    # RNG / seed
    if "rng" in params:
        kwargs["rng"] = rng
    elif "seed" in params:
        # derive a deterministic seed from rng state (stable per run)
        # NOTE: we use an int from rng for a reproducible but independent seed.
        kwargs["seed"] = int(rng.integers(0, 2**31 - 1))
    # theta0 optional (we don't pass it)

    out = simulate_kuramoto_em(**kwargs)

    # Handle return types:
    # A) returns ndarray theta (T,N) or (steps,N)
    # B) returns dataclass-like with .theta, possibly shaped (N,T)
    if isinstance(out, np.ndarray):
        theta = np.asarray(out, dtype=float)
        # If user returns (N,) for some reason, reject
        if theta.ndim != 2:
            raise ValueError(f"simulate_kuramoto_em returned ndarray with shape {theta.shape}, expected 2D.")
        return theta

    # Dataclass / object with theta attribute
    if hasattr(out, "theta"):
        theta = np.asarray(getattr(out, "theta"), dtype=float)
        if theta.ndim != 2:
            raise ValueError(f"simulate_kuramoto_em returned object.theta with shape {theta.shape}, expected 2D.")

        # Common patterns:
        # - (n_osc, n_steps)  -> transpose to (T,N)
        # - (n_steps, n_osc)  -> already (T,N)
        if theta.shape[0] == omega.size and theta.shape[1] == steps:
            return theta.T
        if theta.shape[0] == steps and theta.shape[1] == omega.size:
            return theta
        # Fallback: if first dim equals N, assume (N,T)
        if theta.shape[0] == omega.size:
            return theta.T
        return theta

    raise TypeError("simulate_kuramoto_em returned unsupported type (no ndarray and no .theta).")


# -----------------------------
# Main experiment
# -----------------------------
def run_sweep(protocol: SweepProtocol, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # --- metadata
    timestamp_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    versions: Dict[str, Any] = {
        "timestamp_utc": timestamp_utc,
        "python": _safe_python_version(),
        "numpy": np.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }

    protocol_dict = asdict(protocol)
    protocol_dict["timestamp_utc"] = timestamp_utc

    (outdir / "protocol.json").write_text(json.dumps(protocol_dict, indent=2), encoding="utf-8")
    (outdir / "versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")

    # --- sweep
    K_list = list(map(float, protocol.K_sweep))

    metrics_rows: List[Dict[str, Any]] = []

    for k_index, K in enumerate(K_list):
        r_bars: List[float] = []
        t_syncs: List[float] = []

        for m in range(protocol.M):
            # Deterministic seed per (K,m)
            seed = protocol.base_seed + (k_index * 100_000) + m
            rng = np.random.default_rng(seed)

            omega = rng.normal(protocol.w_mean, protocol.w_std, size=protocol.N).astype(float)

            theta_TN = _call_simulator(
                omega=omega,
                K=K,
                D=protocol.D,
                dt=protocol.dt,
                steps=protocol.steps,
                rng=rng,
            )

            # r(t)
            r_series = _order_parameter_series(theta_TN)

            # steady-state mean of last 25%
            cutoff = int(0.75 * r_series.size)
            r_bar = float(np.mean(r_series[cutoff:]))

            t_sync = _time_to_sync_seconds(
                r_series=r_series,
                dt=protocol.dt,
                threshold=protocol.R_threshold,
                window=protocol.window_size,
            )

            r_bars.append(r_bar)
            t_syncs.append(float(t_sync))

        r_bars_arr = np.asarray(r_bars, dtype=float)
        t_syncs_arr = np.asarray(t_syncs, dtype=float)

        # time-to-sync stats only over successes
        successes = np.isfinite(t_syncs_arr)
        success_frac = float(np.mean(successes)) if t_syncs_arr.size else 0.0

        row: Dict[str, Any] = {
            "K": K,
            "R_mean": float(np.mean(r_bars_arr)),
            "R_q10": float(np.percentile(r_bars_arr, 10)),
            "R_q90": float(np.percentile(r_bars_arr, 90)),
            "Tsync_median_s": "",
            "Tsync_q10_s": "",
            "Tsync_q90_s": "",
            "success_frac": success_frac,
        }

        if np.any(successes):
            t_ok = t_syncs_arr[successes]
            row["Tsync_median_s"] = float(np.median(t_ok))
            row["Tsync_q10_s"] = float(np.percentile(t_ok, 10))
            row["Tsync_q90_s"] = float(np.percentile(t_ok, 90))

        metrics_rows.append(row)

    # --- write CSV
    csv_path = outdir / "kuramoto_metrics.csv"
    _write_metrics_csv(csv_path, metrics_rows)

    # --- plots
    _plot_R_vs_K(outdir / "kuramoto_R_vs_K.png", metrics_rows)
    _plot_Tsync_vs_K(outdir / "kuramoto_Tsync_vs_K.png", metrics_rows)

    print(f"Saved: {csv_path}")
    print(f"Saved: {outdir / 'kuramoto_R_vs_K.png'}")
    print(f"Saved: {outdir / 'kuramoto_Tsync_vs_K.png'}")


def _safe_python_version() -> str:
    import sys

    return sys.version.split()[0]


def _write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "K",
        "R_mean",
        "R_q10",
        "R_q90",
        "Tsync_median_s",
        "Tsync_q10_s",
        "Tsync_q90_s",
        "success_frac",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_R_vs_K(path: Path, rows: List[Dict[str, Any]]) -> None:
    K = np.array([r["K"] for r in rows], dtype=float)
    mean = np.array([r["R_mean"] for r in rows], dtype=float)
    q10 = np.array([r["R_q10"] for r in rows], dtype=float)
    q90 = np.array([r["R_q90"] for r in rows], dtype=float)

    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(K, mean, lw=2)
    plt.fill_between(K, q10, q90, alpha=0.2)
    plt.xlabel("Coupling strength K")
    plt.ylabel(r"Steady-state order parameter $\langle \bar{R} \rangle$")
    plt.title("Transition to Synchronization (Stochastic Kuramoto)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_Tsync_vs_K(path: Path, rows: List[Dict[str, Any]]) -> None:
    Ks: List[float] = []
    med: List[float] = []
    q10: List[float] = []
    q90: List[float] = []

    for r in rows:
        if r["Tsync_median_s"] == "":
            continue
        Ks.append(float(r["K"]))
        med.append(float(r["Tsync_median_s"]))
        q10.append(float(r["Tsync_q10_s"]))
        q90.append(float(r["Tsync_q90_s"]))

    plt.figure(figsize=(8, 5), dpi=300)
    if Ks:
        K = np.array(Ks, dtype=float)
        med_a = np.array(med, dtype=float)
        q10_a = np.array(q10, dtype=float)
        q90_a = np.array(q90, dtype=float)

        plt.plot(K, med_a, marker="o", lw=2)
        plt.fill_between(K, q10_a, q90_a, alpha=0.2)

    plt.xlabel("Coupling strength K")
    plt.ylabel(r"Time to sync $T_{sync}$ (s)")
    plt.title("Response Latency vs Coupling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, required=True, help="Output directory, e.g. results/kuramoto_v3")
    p.add_argument("--N", type=int, default=50)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--D", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--w-mean", type=float, default=0.0)
    p.add_argument("--w-std", type=float, default=0.5)
    p.add_argument("--R-threshold", type=float, default=0.5)
    p.add_argument("--window-size", type=int, default=200)
    p.add_argument(
        "--K-sweep",
        type=str,
        default="0.0,0.1,0.2,0.4,0.8,1.2,1.6,2.0,2.5,3.0",
        help="Comma-separated list of K values.",
    )
    p.add_argument("--base-seed", type=int, default=12345)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    K_sweep = tuple(float(x.strip()) for x in args.K_sweep.split(",") if x.strip() != "")

    protocol = SweepProtocol(
        N=args.N,
        dt=args.dt,
        D=args.D,
        steps=args.steps,
        M=args.M,
        w_mean=args.w_mean,
        w_std=args.w_std,
        K_sweep=K_sweep,
        R_threshold=args.R_threshold,
        window_size=args.window_size,
        base_seed=args.base_seed,
    )

    run_sweep(protocol, outdir)


if __name__ == "__main__":
    main()
