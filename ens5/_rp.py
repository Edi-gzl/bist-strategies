"""Portable replacement for the friend's hard-coded C:\\Research paths.

_RP(r"\\Research_done\\claude_data\\raw") -> <ENS5_ROOT>/Research_done/claude_data/raw
ENS5_ROOT defaults to this folder (repo/ens5). run_ens5.py also creates the
top-level aliases the original layout had (claude_data, Fable-5, INTRADAY_DAILY,
Daily_live -> Research_done/...), so ROOT / 'claude_data' style joins keep working.
"""
import os
from pathlib import Path

BASE = Path(os.environ.get("ENS5_ROOT", Path(__file__).resolve().parent))


def _RP(rest: str = "") -> str:
    """Return a plain string, exactly like the original string literal did."""
    parts = [p for p in rest.replace("/", "\\").split("\\") if p]
    return str(BASE.joinpath(*parts))
