--- a/src/noetiko_convergence/switching.py
+++ b/src/noetiko_convergence/switching.py
@@ -1,6 +1,4 @@
 from __future__ import annotations
 
-from typing import Tuple
-
 import numpy as np


@dataclass(frozen=True)
class SwitchingStats:
    tau_L: float
    tau_H: float
    rate_L_to_H: float
    rate_H_to_L: float
    n_transitions: int


def switching_statistics(
    r_t: np.ndarray,
    fs: float,
    r_L: float,
    r_H: float,
) -> SwitchingStats:
    """Estimate dwell times and transition rates between low/high r-regimes.

    Parameters
    ----------
    r_t : np.ndarray
        Time series of order parameter r(t).
    fs : float
        Sampling rate (Hz) of r(t).
    r_L, r_H : float
        Low/high thresholds with r_L < r_H.
    """
    r_t = np.asarray(r_t, dtype=float)
    if r_L >= r_H:
        raise ValueError("Require r_L < r_H.")
    if fs <= 0:
        raise ValueError("fs must be positive.")

    # state: -1 low, +1 high, 0 undecided (between thresholds)
    state = np.zeros_like(r_t, dtype=int)
    state[r_t <= r_L] = -1
    state[r_t >= r_H] = +1

    # fill undecided by last state (hysteresis) to avoid chattering
    for i in range(1, len(state)):
        if state[i] == 0:
            state[i] = state[i-1]
    # ignore leading zeros
    if state[0] == 0:
        first = np.nonzero(state)[0]
        if len(first) == 0:
            raise ValueError("No samples cross either threshold.")
        state[:first[0]] = state[first[0]]

    # transitions
    changes = np.nonzero(np.diff(state))[0]
    n_trans = int(len(changes))
    # dwell durations
    idx = np.r_[0, changes+1, len(state)]
    dwells = np.diff(idx) / fs
    dwell_states = state[idx[:-1]]

    tau_L = float(np.mean(dwells[dwell_states == -1])) if np.any(dwell_states == -1) else float("nan")
    tau_H = float(np.mean(dwells[dwell_states == +1])) if np.any(dwell_states == +1) else float("nan")

    # rates ~ transitions out of state / total time in state
    time_L = float(np.sum(dwells[dwell_states == -1]))
    time_H = float(np.sum(dwells[dwell_states == +1]))
    out_L = int(np.sum((dwell_states[:-1] == -1) & (dwell_states[1:] == +1)))
    out_H = int(np.sum((dwell_states[:-1] == +1) & (dwell_states[1:] == -1)))
    rate_LH = out_L / time_L if time_L > 0 else float("nan")
    rate_HL = out_H / time_H if time_H > 0 else float("nan")

    return SwitchingStats(
        tau_L=tau_L,
        tau_H=tau_H,
        rate_L_to_H=float(rate_LH),
        rate_H_to_L=float(rate_HL),
        n_transitions=n_trans,
    )
