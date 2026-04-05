"""Artifact writers with immutability guarantees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from art.errors import ArtError


def _assert_writable(path: Path) -> None:
    if path.exists():
        raise ArtError(f"Artifact already exists (immutable): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    _assert_writable(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    _assert_writable(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def write_text(path: Path, content: str) -> None:
    _assert_writable(path)
    path.write_text(content, encoding="utf-8")
