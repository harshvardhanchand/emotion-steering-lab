"""Local import shim for src-layout package during editable/dev runs.

This allows `uv run art ...` to work from the repository root even when
editable path files are skipped by the interpreter.
"""

from __future__ import annotations

from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
_src_art = _repo_root / "src" / "art"

# Expose the real package modules from src/art as this package path.
__path__ = [str(_src_art)]  # type: ignore[name-defined]
