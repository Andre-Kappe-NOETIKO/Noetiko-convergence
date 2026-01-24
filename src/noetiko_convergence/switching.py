from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SwitchingStats:
    tau_L: float
    tau_H: float
    r_LH: float
    r_HL: float
    n_L: int
    n_H: int


def estimate_switching_stats(
    r_t: np.ndarray,
    r_L: float,
    r_H: float,
    dt: float,
) -> SwitchingStats:
    """
    Estimate dwell times and transition rates between low/high coordination regimes.

    A point is labeled:
      - low if r <= r_L
      - high if r >= r_H
      - otherwise unlabeled (ignored for dwell estimation)

    Returns:
        SwitchingStats with mean dwell times and empirical transition rates.
    """
    r_t = np.asarray(r_t, dtype=float)
    if r_t.ndim != 1:
        raise ValueError("r_t must be 1D")
    if not (0.0 <= r_L < r_H <= 1.0):
        raise ValueError("Require 0 <= r_L < r_H <= 1")

    labels = np.full(r_t.shape[0], fill_value=0, dtype=int)
    labels[r_t <= r_L] = -1
    labels[r_t >= r_H] = +1

    dwells_L = []
    dwells_H = []

    i = 0
    while i < labels.size:
        if labels[i] == 0:
            i += 1
            continue

        state = labels[i]
        j = i
        while j < labels.size and labels[j] == state:
            j += 1

        dwell = (j - i) * dt
        if state == -1:
            dwells_L.append(dwell)
        else:
            dwells_H.append(dwell)

        i = j

    tau_L = float(np.mean(dwells_L)) if dwells_L else float("nan")
    tau_H = float(np.mean(dwells_H)) if dwells_H else float("nan")

    # transitions between labeled states
    trans_LH = 0
    trans_HL = 0
    prev = 0
    for s in labels:
        if s == 0:
            continue
        if prev == -1 and s == +1:
            trans_LH += 1
        if prev == +1 and s == -1:
            trans_HL += 1
        prev = s

    total_L = sum(dwells_L) if dwells_L else 0.0
    total_H = sum(dwells_H) if dwells_H else 0.0

    r_LH = trans_LH / total_L if total_L > 0.0 else float("nan")
    r_HL = trans_HL / total_H if total_H > 0.0 else float("nan")

    return SwitchingStats(
        tau_L=tau_L,
        tau_H=tau_H,
        r_LH=r_LH,
        r_HL=r_HL,
        n_L=len(dwells_L),
        n_H=len(dwells_H),
    )
