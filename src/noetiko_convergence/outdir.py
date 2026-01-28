from __future__ import annotations

import os
from pathlib import Path


def get_outdir(default: str | Path = "results") -> Path:
    """
    Single source of truth for output directory.

    Priority:
      1) env var NOETIKO_OUTDIR
      2) default argument
    """
    raw = os.environ.get("NOETIKO_OUTDIR", str(default))
    p = Path(raw).expanduser()
    try:
        return p.resolve()
    except FileNotFoundError:
        # If it doesn't exist yet, resolve may fail on some systems
        return p.absolute()


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
