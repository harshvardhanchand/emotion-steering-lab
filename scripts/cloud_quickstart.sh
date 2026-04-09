#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8501}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run this script from the repository root (pyproject.toml not found)." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  echo "uv not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv install appears to have failed. Try: source \$HOME/.local/bin/env" >&2
  exit 1
fi
export PATH="$(dirname "$UV_BIN"):$PATH"
if [[ "$(id -u)" == "0" && -w /usr/local/bin ]]; then
  ln -sf "$UV_BIN" /usr/local/bin/uv || true
  [[ -x "$HOME/.local/bin/uvx" ]] && ln -sf "$HOME/.local/bin/uvx" /usr/local/bin/uvx || true
fi
echo "Using uv at: $UV_BIN"

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="${STREAMLIT_SERVER_FILE_WATCHER_TYPE:-none}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}"

echo "Syncing dependencies..."
"$UV_BIN" sync --all-extras --no-editable

echo "Starting Emotion Steering Studio at ${HOST}:${PORT}..."
"$UV_BIN" run python -m streamlit run src/art/ui_app.py \
  --server.address "${HOST}" \
  --server.port "${PORT}" \
  --server.headless true \
  --server.fileWatcherType none \
  --server.runOnSave false \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.enableWebsocketCompression false \
  --browser.gatherUsageStats false \
  --logger.level error
