"""Noetiko Convergence: reproducibility toolkit for Papers I–III.

Design principle: conservative, operational, falsifiable. No semantic claims.

Import submodules explicitly, e.g.:
- from noetiko_convergence.phase import hilbert_phase
- from noetiko_convergence.kuramoto import simulate_kuramoto_em
- from noetiko_convergence.surrogates import phase_randomization_surrogate
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("noetiko-convergence")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
