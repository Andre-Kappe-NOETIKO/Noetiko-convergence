"""Noetiko Convergence: reproducibility toolkit for Papers I–III.

This package implements operational measurement-to-model pipelines:
phase extraction, quality gates, surrogate controls, coordination metrics,
Kuramoto-type sanity simulations, and switching/barrier inference helpers.

Design principle: conservative, operational, falsifiable. No semantic claims.
"""

from .metrics import order_parameter
from .phase import hilbert_phase
from .surrogates import phase_randomization_surrogate, time_shift_surrogate
from .kuramoto import simulate_kuramoto_em

__all__ = [
    "order_parameter",
    "hilbert_phase",
    "phase_randomization_surrogate",
    "time_shift_surrogate",
    "simulate_kuramoto_em",
]
