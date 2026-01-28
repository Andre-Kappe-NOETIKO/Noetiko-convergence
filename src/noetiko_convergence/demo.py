python - <<'PY'
from pathlib import Path

content = """from __future__ import annotations

import argparse
import json
from pathlib import Path as _Path
from typing import Any

import numpy as np

from .kuramoto import order_parameter, simulate_kuramoto_global
from .phase import compute_phases_multichannel
from .quality_gates import consistency_gate, robustness_gate, spectral_gate
from .surrogates import phase_randomization_multichannel, time_shift_surrogate


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
    \"\"\"Proof-of-protocol demo (non-interpretive).

    Pipeline:
    - simulate multichannel oscillatory data (stand-in)
    - extract phases
    - apply quality gates
    - compute r(t) = |R(t)|
    - compare against mandatory surrogates

    Returns:
        dict with summary metrics and (lightweight) arrays for plotting.
    \"\"\"
    rng = np.random.default_rng(seed)

    # 1) synthetic phases via noisy Kuramoto
    omega = rng.normal(loc=0.0, scale=0.8, size=int(n_osc))
    sim = simulate_kuramoto_global(
        omega,
        K=float(K),
        D=float(D),
        dt=float(dt),
        steps=int(steps),
        seed=int(seed),
    )
    theta = sim.theta  # shape (n_osc, n_steps)

    # 2) synthetic observables x_i(t) from phases
    X = np.cos(theta) + 0.12 * rng.standard_normal(size=theta.shape)

    # 3) extract phases
    phases, amp = compute_phases_multichannel(X, method="hilbert", unwrap=True)

    # 4) gates
    g1 = bool(spectral_gate(X, fs=float(fs), f_lo=float(f_lo), f_hi=float(f_hi)))
    g2 = bool(robustness_gate(phases, amp=amp))
    g3 = bool(consistency_gate(phases))
    passed = float(g1 and g2 and g3)

    # 5) order parameter and r(t)
    R_t = order_parameter(phases, axis=0)   # complex, shape (n_steps,)
    r_t = np.abs(R_t)

    # 6) surrogates: phase-randomization
    X_pr = phase_randomization_multichannel(X, rng=rng)
    phases_pr, _ = compute_phases_multichannel(X_pr, method="hilbert", unwrap=True)
    r_pr = np.abs(order_parameter(phases_pr, axis=0))

    # 7) surrogates: time-shift
    X_ts = time_shift_surrogate(X, shift=int(shift))
    phases_ts, _ = compute_phases_multichannel(X_ts, method="hilbert", unwrap=True)
    r_ts = np.abs(order_parameter(phases_ts, axis=0))

    # 8) simple separation metrics
    mean_r = float(np.mean(r_t))
    mean_r_pr = float(np.mean(r_pr))
    mean_r_ts = float(np.mean(r_ts))

    return {
        "passed_gates": passed,
        "gate_spectral": float(g1),
        "gate_robustness": float(g2),
        "gate_consistency": float(g3),
        "mean_r": mean_r,
        "mean_r_phase_rand": mean_r_pr,
        "mean_r_time_shift": mean_r_ts,
        "sep_phase_rand": float(mean_r - mean_r_pr),
        "sep_time_shift": float(mean_r - mean_r_ts),
        # for plotting/debug (small)
        "r_t": r_t.astype(float),
        "r_pr": r_pr.astype(float),
        "r_ts": r_ts.astype(float),
    }


def _write_summary(out_dir: _Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON (arrays -> lists)
    json_ready: dict[str, Any] = {}
    for k, v in summary.items():
        if isinstance(v, np.ndarray):
            json_ready[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            json_ready[k] = v.item()
        else:
            json_ready[k] = v

    (out_dir / "summary.json").write_text(
        json.dumps(json_ready, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # human-readable text
    lines = [
        f"passed_gates: {summary['passed_gates']}",
        f"gate_spectral: {summary['gate_spectral']}",
        f"gate_robustness: {summary['gate_robustness']}",
        f"gate_consistency: {summary['gate_consistency']}",
        f"mean_r: {summary['mean_r']}",
        f"mean_r_phase_rand: {summary['mean_r_phase_rand']}",
        f"mean_r_time_shift: {summary['mean_r_time_shift']}",
        f"sep_phase_rand: {summary['sep_phase_rand']}",
        f"sep_time_shift: {summary['sep_time_shift']}",
    ]
    (out_dir / "summary.txt").write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def _plot(out_dir: _Path, r_t: np.ndarray, r_pr: np.ndarray, r_ts: np.ndarray) -> None:
    # Import lazily so demo still works headless
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.arange(r_t.size, dtype=float)

    plt.figure()
    plt.plot(t, r_t, label="r(t)")
    plt.plot(t, r_pr, label="r_phase_rand(t)")
    plt.plot(t, r_ts, label="r_time_shift(t)")
    plt.xlabel("step")
    plt.ylabel("r = |R|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "demo_r_surrogates.png", dpi=160)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="noetiko_convergence demo (writes outputs into --out)")
    p.add_argument("--out", type=str, default="results/demo", help="Output directory (created if missing).")
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
    p.add_argument("--no-plot", action="store_true", help="Do not write demo_r_surrogates.png")
    args = p.parse_args(argv)

    out_dir = _Path(args.out)

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

    _write_summary(out_dir, summary)

    if not args.no_plot:
        _plot(out_dir, np.asarray(summary["r_t"]), np.asarray(summary["r_pr"]), np.asarray(summary["r_ts"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

path = Path("src/noetiko_convergence/demo.py")
path.write_text(content, encoding="utf-8")
print("WROTE", path, "bytes=", len(content.encode("utf-8")))
PY
