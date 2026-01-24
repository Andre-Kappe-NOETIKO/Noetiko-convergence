from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import json


@dataclass
class ReportingChecklist:
    observables: str
    phase_map: str
    quality_gates: str
    coordination_metrics: str
    surrogate_controls: str
    perturbation_sweeps: str
    uncertainty: str
    null_outcomes: str

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, ensure_ascii=False)


DEFAULT_CHECKLIST = ReportingChecklist(
    observables="definition of x_i(t), units, sampling rate, preprocessing (filters, windowing)",
    phase_map="exact method Φ, parameters, unwrap rule, robustness check",
    quality_gates="spectral/robustness/consistency gates; rejection rate across windows",
    coordination_metrics="r(t) and (optional) Q(D); auxiliary discriminators when relevant",
    surrogate_controls="phase randomization and time-shift surrogates (optional architecture null)",
    perturbation_sweeps="coupling proxy and/or noise proxy D; sweep protocol and bandwidth",
    uncertainty="confidence intervals across windows/runs; sensitivity to mild Φ variations",
    null_outcomes="explicit reporting of failed gates or retained nulls (Prediction sets A–D)",
)
