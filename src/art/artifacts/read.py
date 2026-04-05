"""Artifact readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from art.errors import ArtError


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtError(f"Missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ArtError(f"Missing artifact: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows
