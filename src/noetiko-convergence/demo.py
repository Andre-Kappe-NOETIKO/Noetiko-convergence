from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from .phase import hilbert_phase
from .metrics import order_parameter
from .surrogates import phase_randomization_surrogate, time_shift_surrogate
from .quality_gates import windowed_snr_gate


def _synthetic_multichannel(fs: float, seconds: float, n_ch: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    x = np.zeros((n_ch, t.size), dtype=float)
    # shared oscillation + channel phase offsets + independent noise
    base_phase = rng.uniform(0, 2*np.pi)
    for i in range(n_ch):
        phi = base_phase + rng.uniform(-0.6, 0.6)
        x[i] = np.sin(2*np.pi*f0*t + phi) + 0.4*rng.standard_normal(size=t.size)
    return x


def run_demo(out_dir: Path, fs: float = 250.0, seconds: float = 20.0, band: Tuple[float, float] = (6.0, 10.0)) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    x = _synthetic_multichannel(fs=fs, seconds=seconds, n_ch=16, f0=8.0, rng=rng)
    res = hilbert_phase(x, unwrap=True, fs_hz=fs, band_hz=band)

    # Quality gate using amplitude proxy
    gate = windowed_snr_gate(res.amplitude, window_size=int(fs*1.0), amp_threshold=float(np.median(res.amplitude) * 0.6))

    op = order_parameter(res.phase, axis=0)
    r = op.r

    # Surrogates (compute r for each)
    x_pr = phase_randomization_surrogate(x, rng=rng)
    pr_res = hilbert_phase(x_pr, unwrap=True, fs_hz=fs, band_hz=band)
    r_pr = order_parameter(pr_res.phase, axis=0).r

    x_ts = time_shift_surrogate(x, min_shift=int(fs*2.0), rng=rng)
    ts_res = hilbert_phase(x_ts, unwrap=True, fs_hz=fs, band_hz=band)
    r_ts = order_parameter(ts_res.phase, axis=0).r

    # Plot
    t = np.arange(r.size) / fs
    plt.figure()
    plt.plot(t, r, label="observed r(t)")
    plt.plot(t, r_pr, label="phase-rand surrogate")
    plt.plot(t, r_ts, label="time-shift surrogate")
    plt.xlabel("time (s)")
    plt.ylabel("r(t)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "demo_r_surrogates.png", dpi=200)

    # Save gate summary
    (out_dir / "gate_summary.txt").write_text(
        f"Windows passed: {gate.passed.sum()}/{gate.passed.size}\nRejection rate: {gate.rejection_rate:.3f}\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Noetiko Convergence proof-of-protocol demo (synthetic).")
    ap.add_argument("--out", default="results/demo", help="Output directory")
    args = ap.parse_args()
    run_demo(Path(args.out))


if __name__ == "__main__":
    main()
