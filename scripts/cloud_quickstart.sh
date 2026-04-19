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
if [[ -z "$UV_BIN" || ! -x "$UV_BIN" ]]; then
  UV_BIN=""
  while IFS= read -r candidate; do
    if [[ -x "$candidate" ]]; then
      UV_BIN="$candidate"
      break
    fi
  done < <(type -aP uv 2>/dev/null || true)
fi
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv install appears to have failed. Try: source \$HOME/.local/bin/env" >&2
  exit 1
fi
export PATH="$(dirname "$UV_BIN"):$PATH"
if [[ "$(id -u)" == "0" && -w /usr/local/bin ]]; then
  if [[ "$UV_BIN" != "/usr/local/bin/uv" ]]; then
    ln -sf "$UV_BIN" /usr/local/bin/uv || true
  fi
  if [[ -x "$HOME/.local/bin/uvx" && "$HOME/.local/bin/uvx" != "/usr/local/bin/uvx" ]]; then
    ln -sf "$HOME/.local/bin/uvx" /usr/local/bin/uvx || true
  fi
fi
echo "Using uv at: $UV_BIN"

export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
export TRANSFORMERS_NO_ADVISORY_WARNINGS="${TRANSFORMERS_NO_ADVISORY_WARNINGS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export STREAMLIT_SERVER_FILE_WATCHER_TYPE="${STREAMLIT_SERVER_FILE_WATCHER_TYPE:-none}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="${STREAMLIT_BROWSER_GATHER_USAGE_STATS:-false}"

echo "Syncing dependencies..."
"$UV_BIN" sync --all-extras --no-editable --frozen

CUDA_VERSION_RAW=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VERSION_RAW="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n 1 || true)"
fi

TORCH_CHANNEL=""
if [[ -n "$CUDA_VERSION_RAW" ]]; then
  CUDA_MAJOR="${CUDA_VERSION_RAW%%.*}"
  CUDA_MINOR="${CUDA_VERSION_RAW#*.}"
  CUDA_MINOR="${CUDA_MINOR%%[^0-9]*}"
  if [[ -n "$CUDA_MAJOR" && -n "$CUDA_MINOR" ]]; then
    TORCH_CHANNEL="cu${CUDA_MAJOR}${CUDA_MINOR}"
  fi
fi

if [[ -n "$TORCH_CHANNEL" ]]; then
  CURRENT_TORCH_CHANNEL="$("$UV_BIN" run python -c "import torch; v=getattr(torch.version, 'cuda', None); print(('cu' + ''.join(v.split('.')[:2])) if v else '')" 2>/dev/null || true)"
  if [[ "$CURRENT_TORCH_CHANNEL" != "$TORCH_CHANNEL" ]]; then
    echo "Detected CUDA ${CUDA_VERSION_RAW}; aligning PyTorch wheels to ${TORCH_CHANNEL}..."
    "$UV_BIN" run python -m ensurepip --upgrade >/dev/null 2>&1 || true
    if ! "$UV_BIN" run python -m pip install --upgrade --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}" torch torchvision torchaudio; then
      echo "Warning: Failed to align PyTorch wheels for ${TORCH_CHANNEL}. Continuing with current install." >&2
    fi
  fi
fi

echo "Ensuring hf_transfer is available in the project environment..."
if "$UV_BIN" run python -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
else
  "$UV_BIN" run python -m ensurepip --upgrade >/dev/null 2>&1 || true
  if "$UV_BIN" run python -m pip install hf_transfer; then
    export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
  else
    echo "Warning: hf_transfer install failed; continuing with HF_HUB_ENABLE_HF_TRANSFER=0." >&2
    export HF_HUB_ENABLE_HF_TRANSFER=0
  fi
fi

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
