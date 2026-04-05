# Contributing

Thanks for your interest in improving Emotion Steering Lab.

## Development Setup

1. Install Python 3.11 and `uv`.
2. Install dependencies:

```bash
uv sync --all-extras
```

3. Run checks before opening a PR:

```bash
uv run python scripts/check_contracts.py
uv run pytest -q
```

## Project Standards

- Keep schema contracts backward-compatible unless explicitly versioned.
- Add or update tests for behavior changes.
- Keep CLI errors explicit and non-zero on validation failures.
- Preserve artifact immutability semantics.

## Pull Request Checklist

- [ ] Tests pass locally (`uv run pytest -q`).
- [ ] Contract checks pass (`uv run python scripts/check_contracts.py`).
- [ ] Any schema-related changes include migration or version notes.
- [ ] Docs updated for new flags/commands/artifacts.
- [ ] Changelog entry added under `Unreleased`.

## Commit and PR Guidance

- Use focused commits with clear messages.
- Prefer small PRs with a single objective.
- Include sample command output when changing CLI behavior.
