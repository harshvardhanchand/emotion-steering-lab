#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-user}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://docs.astral.sh/uv/" >&2
  exit 1
fi

case "$MODE" in
  user)
    echo "[setup] Installing stable user environment (non-editable)..."
    uv sync --all-extras --no-editable
    ;;
  dev)
    echo "[setup] Installing editable dev environment..."
    uv sync --all-extras
    ;;
  *)
    echo "Usage: ./scripts/setup.sh [user|dev]" >&2
    exit 1
    ;;
esac

echo
echo "[setup] Done."
echo "[setup] Launch UI: uv run art ui"
echo "[setup] CLI help:  uv run art --help"
