# Emotion Steering Lab

Inference-time steering framework with a method-faithful pipeline inspired by Anthropic's emotion concepts work:

1. Generate synthetic probe data
2. Train probe vectors
3. Run steering sweeps
4. Save reusable steering profiles
5. Explore effects in a small UI chat sandbox

## Quickstart

```bash
# recommended install for end users (avoids editable-install path issues)
uv sync --all-extras --no-editable
# or: ./scripts/setup.sh user

uv run art data generate --backend transformers --model-id Qwen/Qwen2.5-0.5B-Instruct --emotion happy --emotion sad --emotion desperate --out data/probe_data.jsonl
uv run art probe train --probe-data data/probe_data.jsonl --backend transformers --model-id Qwen/Qwen2.5-0.5B-Instruct --out artifacts/probe_artifact.json
uv run art probe diagnose --probe-artifact artifacts/probe_artifact.json --cases data/probe_data.jsonl --backend transformers --model-id Qwen/Qwen2.5-0.5B-Instruct --out reports/diagnosis.jsonl
uv run art steer run --probe-artifact artifacts/probe_artifact.json --probe-name calm --cases data/probe_data.jsonl --backend transformers --model-id Qwen/Qwen2.5-0.5B-Instruct --save-profile profiles/default.json
uv run art steer run --probe-artifact artifacts/probe_artifact.json --cases data/probe_data.jsonl --backend transformers --model-id Qwen/Qwen2.5-0.5B-Instruct --profile profiles/default.json
uv run art steer sweep --probe-artifact artifacts/probe_artifact.json --probe-name calm --cases data/probe_data.jsonl --backend transformers --model-id Qwen/Qwen2.5-0.5B-Instruct
```

For contributors using editable installs:

```bash
./scripts/setup.sh dev
```

For local smoke tests without heavy model deps:

```bash
uv run art data generate --backend mock --model-id mock/model --tokenizer-id mock/tokenizer --emotion happy --emotion sad --out data/probe_data.jsonl
uv run art probe train --probe-data data/probe_data.jsonl --backend mock --model-id mock/model --tokenizer-id mock/tokenizer --out artifacts/probe_artifact.json
uv run art probe diagnose --probe-artifact artifacts/probe_artifact.json --cases data/probe_data.jsonl --backend mock --model-id mock/model --tokenizer-id mock/tokenizer --out reports/diagnosis.jsonl
uv run art steer run --probe-artifact artifacts/probe_artifact.json --probe-name calm --cases data/probe_data.jsonl --backend mock --model-id mock/model --tokenizer-id mock/tokenizer
uv run art steer sweep --probe-artifact artifacts/probe_artifact.json --probe-name calm --cases data/probe_data.jsonl --backend mock --model-id mock/model --tokenizer-id mock/tokenizer
```

## UI

```bash
uv sync --extra ui --extra hf --no-editable
# or: ./scripts/setup.sh user
uv run art ui
```

If you already synced without `--no-editable`, re-sync once:

```bash
uv sync --all-extras --no-editable
```

If you still need a fallback launcher:

```bash
uv run python -m art.cli ui
```

`art ui` now launches Streamlit with watcher disabled and error-level logs by default to suppress optional `torchvision` import noise from `transformers`.

If you use editable installs (`./scripts/setup.sh dev`) on macOS and hit `ModuleNotFoundError: No module named 'art'`, re-run setup once:

```bash
./scripts/setup.sh dev
```

The setup script now clears macOS hidden flags on venv `.pth` files so Python can load the local package path correctly.

## Run On Cloud VM In 5 Commands

```bash
git clone <YOUR_REPO_URL>
cd emotion-steering-lab
chmod +x scripts/cloud_quickstart.sh
./scripts/cloud_quickstart.sh
ssh -N -L 8501:127.0.0.1:8501 <user>@<vm-ip>
```

Then open `http://127.0.0.1:8501`.

UI includes:

- unified build modes: `Generate + Train`, `Train from Existing Data`, `Load Existing Artifact`
- preflight checks with path immutability/runtime warnings
- background jobs with live progress and `Stop` cancellation
- session build history with one-click artifact reload
- faster `Generate + Train`: single backend/model instance is reused across generation and training
- faster `Generate + Train` data build: generates `story` + `neutral_dialogue` rows for probe fitting (emotional dialogues can be generated separately for eval)
- batched text generation during data build when backend supports `generate_batch`
- generation cache (`cache/generation`) to skip repeat prompt generations across reruns; configurable via CLI/UI
- paper-faithful training guards (`>=2` emotions and required `neutral_dialogue` rows)
- faster probe fitting: activation extraction uses `story` + `neutral_dialogue` rows; emotional dialogues remain available for eval/diagnosis
- batched hidden-state extraction during training when backend supports it
- activation cache for probe training (`cache/activations`) to speed reruns; configurable with `art probe train --activation-cache-dir` or `--no-activation-cache`
- resumable `Generate + Train`: if `probe_data.jsonl` already exists and artifact does not, UI resumes from existing data and writes `<probe_artifact_stem>.checkpoint.json`
- multi-emotion steering sliders with profile save/load
- live chat sandbox showing baseline vs steered outputs with score/severity deltas

## Commands

- `art data generate`
- `art probe train`
- `art probe diagnose`
- `art steer run`
- `art steer sweep`
- `art steer alpha-grid`
- `art ui`

## Schemas

- `schemas/probe_data.schema.json`
- `schemas/probe_artifact.schema.json`
- `schemas/steering_results.schema.json`
- `schemas/steering_profile.schema.json`
- `schemas/diagnosis.schema.json`

## Reproducibility

- Every command that writes primary artifacts also writes an immutable sidecar manifest: `*.meta.json`
- Manifests include frozen config, config hash, input hashes, and output hashes
- Generated `probe_data` rows include generation attempt/QC provenance hashes

## License

Apache-2.0. See `LICENSE`.
