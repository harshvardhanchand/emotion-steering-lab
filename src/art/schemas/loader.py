"""Load JSON schemas from the project-level schemas directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from art.constants import project_root
from art.errors import ArtError


def schema_path(schema_filename: str) -> Path:
    path = project_root() / "schemas" / schema_filename
    if not path.exists():
        raise ArtError(f"Schema file not found: {path}")
    return path


def load_schema(schema_filename: str) -> dict[str, Any]:
    return json.loads(schema_path(schema_filename).read_text(encoding="utf-8"))
