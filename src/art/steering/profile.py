"""Steering profile helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from art.constants import SCHEMA_VERSION


def build_profile(
    *,
    profile_id: str,
    model_id: str,
    probe_name: str,
    layers: list[int],
    alpha: float,
    scope: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "probe_name": probe_name,
        "layers": layers,
        "alpha": alpha,
        "scope": scope,
        "enabled": True,
    }
