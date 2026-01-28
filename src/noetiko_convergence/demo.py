from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .kuramoto import order_parameter, simulate_kuramoto_global
from .phase import compute_phases_multichannel
from .quality_gates import consistency_gate, robustness_gate, spectral_gate
from .surrogates import phase_randomization_multichannel, time_shift_surrogate


def _ensure_theta(sim: Any) -> np.ndarray:
    """Accept either ndarray or a dataclass-like result with attribute `.theta`."""
    if hasattr(sim, "theta"):
        return np.asarray(sim.theta, dtype=float)
    return np.asarray(sim, dtype=float)


def run_demo(
    *,
    seed: int = 0,
    n_osc: int = 12,
    steps: int = 2500,
    K: float = 1.7,
    D: float = 0.08,
    dt: float = 0.02,
    fs: float = 50.0,
    f_lo: float = 0.5,
    f_hi: float = 10.0,
    shift: int = 250,
) -> dict[str, Any]:
    """
    Proof-of-protocol demo:
    - simulate multichannel oscillatory data (stand-in)
    - extract phases
    - apply quality gates
    - compute r(t)
    - compare against mandatory surrogates
    """

    rng = np.random.default_rng(seed)

    omega = rng.normal(loc=0.0, scale=0.8, size=int(n_osc))
    sim = simulate_kuramoto_global(omega, K=float(K), D=float(D), dt=float(dt), steps=int(steps), seed=int(seed))
    theta = _ensure_theta(sim)  # (n_osc, n_steps)

    # synthetic observables x_i(t)
    X = np.cos(theta) + 0.12 * rng.standard_normal(size=theta.shape)

    # phases + amplitude
    phases, amp = compute_phases_multichannel(X, method="hilbert", unwrap=True)

    # gates
    g1 = bool(spectral_gate(X, fs=float(fs), f_lo=float(f_lo), f_hi=float(f_hi)))
    g2 = bool(robustness_gate(phases, amp=amp))
    g3 = bool(consistency_gate(phases))
    passed = float(g1 and g2 and g3)

    # order parameter
    r_t = np.abs(order_parameter(phases, axis=0))  # -> (n_steps,)

    # surrogates
    X_pr = phase_randomization_multichannel(X, rng=rng)
    phases_pr, _ = compute_phases_multichannel(X_pr, method="hilbert", unwrap=True)
    r_pr = np.abs(order_parameter(phases_pr, axis=0))

    X_ts = time_shift_surrogate(X, shift=int(shift))
    phases_ts, _ = compute_phases_multichannel(X_ts, method="hilbert", unwrap=True)
    r_ts = np.abs(order_parameter(phases_ts, axis=0))

    # metrics
    mean_r = float(np.mean(r_t))
    mean_r_pr = float(np.mean(r_pr))
    mean_r_ts = float(np.mean(r_ts))

    return {
        "passed_gates": passed,
        "gates": {"spectral": g1, "robustness": g2, "consistency": g3},
        "params": {
            "seed": int(seed),
            "n_osc": int(n_osc),
            "steps": int(steps),
            "K": float(K),
            "D": float(D),
            "dt": float(dt),
            "fs": float(fs),
            "f_lo": float(f_lo),
            "f_hi": float(f_hi),
            "shift": int(shift),
        },
        "mean_r": mean_r,
        "mean_r_phase_rand": mean_r_pr,
        "mean_r_time_shift": mean_r_ts,
        "sep_phase_rand": float(mean_r - mean_r_pr),
        "sep_time_shift": float(mean_r - mean_r_ts),
        # keep JSON small: store short snippets only
        "r_snippet": {
            "r_t": [float(x) for x in r_t[:200]],
            "r_pr": [float(x) for x in r_pr[:200]],
            "r_ts": [float(x) for x in r_ts[:200]],
        },
    }


def _write_outputs(outdir: Path, summary: dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # text summary (human-readable, minimal)
    txt = [
        f"passed_gates: {summary['passed_gates']}",
        f"gates: {summary['gates']}",
        f"mean_r: {summary['mean_r']:.6f}",
        f"mean_r_phase_rand: {summary['mean_r_phase_rand']:.6f}",
        f"mean_r_time_shift: {summary['mean_r_time_shift']:.6f}",
        f"sep_phase_rand: {summary['sep_phase_rand']:.6f}",
        f"sep_time_shift: {summary['sep_time_shift']:.6f}",
    ]
    (outdir / "summary.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")

    # JSON summary
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # plot
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        r = summary["r_snippet"]
        x = np.arange(len(r["r_t"]))
        plt.figure()
        plt.plot(x, r["r_t"], label="r(t)")
        plt.plot(x, r["r_pr"], label="phase-rand")
        plt.plot(x, r["r_ts"], label="time-shift")
        plt.xlabel("t index (snippet)")
        plt.ylabel("|R|")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "demo_r_surrogates.png", dpi=150)
        plt.close()
    except Exception as e:  # pragma: no cover
        # still succeed even if matplotlib backend is weird
        (outdir / "plot_error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m noetiko_convergence.demo", description="Run Noetiko convergence demo.")
    p.add_argument("--out", type=Path, default=Path("results/demo"), help="Output directory.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-osc", type=int, default=12)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--K", type=float, default=1.7)
    p.add_argument("--D", type=float, default=0.08)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--f-lo", type=float, default=0.5)
    p.add_argument("--f-hi", type=float, default=10.0)
    p.add_argument("--shift", type=int, default=250)

    args = p.parse_args(argv)

    summary = run_demo(
        seed=args.seed,
        n_osc=args.n_osc,
        steps=args.steps,
        K=args.K,
        D=args.D,
        dt=args.dt,
        fs=args.fs,
        f_lo=args.f_lo,
        f_hi=args.f_hi,
        shift=args.shift,
    )
    _write_outputs(args.out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
