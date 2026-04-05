"""Path and run-id helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from art.constants import runs_root
from art.errors import ArtError


def make_run_id(now: datetime | None = None) -> str:
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    return f"{ts}_{suffix}"


def get_run_dir(run_id: str) -> Path:
    return runs_root() / run_id


def create_run_dir(run_id: str) -> Path:
    root = runs_root()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / run_id
    if run_dir.exists():
        raise ArtError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def ensure_existing_run_dir(run_id: str) -> Path:
    run_dir = get_run_dir(run_id)
    if not run_dir.exists() or not run_dir.is_dir():
        raise ArtError(f"Run directory not found: {run_dir}")
    return run_dir
