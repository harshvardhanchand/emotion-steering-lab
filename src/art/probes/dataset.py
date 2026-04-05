"""Probe dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from art.artifacts.read import read_jsonl
from art.errors import ArtError


def load_probe_data(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ArtError(f"Probe data not found: {path}")
    rows = read_jsonl(path)
    if not rows:
        raise ArtError("Probe dataset is empty")
    return rows
