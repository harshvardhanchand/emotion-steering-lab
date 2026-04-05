from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def set_project_root_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ART_PROJECT_ROOT", str(Path.cwd()))
