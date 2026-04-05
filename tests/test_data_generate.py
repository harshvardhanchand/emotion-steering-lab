from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from art.cli import app
from art.data.generate import DataGenConfig, generate_probe_data
from art.schemas.validator import validate_documents


def test_data_generate_cli(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data",
            "generate",
            "--out",
            "data/probe_data.jsonl",
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--topic",
            "A test topic 1",
            "--topic",
            "A test topic 2",
            "--emotion",
            "afraid",
            "--emotion",
            "amazed",
        ],
    )
    assert result.exit_code == 0, result.output

    out = tmp_path / "data/probe_data.jsonl"
    assert out.exists()
    assert (tmp_path / "data/probe_data.jsonl.meta.json").exists()

    rows = [__import__("json").loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) > 0
    validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")
    assert all(bool(r.get("metadata", {}).get("qc_passed")) for r in rows)
    assert all(int(r.get("metadata", {}).get("generation_attempt", 0)) >= 1 for r in rows)


def test_data_generate_uses_injected_backend(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _InlineBackend:
        backend_name = "mock"
        model_id = "mock/model"
        tokenizer_id = "mock/tokenizer"

        def generate(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            steering=None,
            max_length: int = 2048,
        ) -> tuple[str, int]:
            del max_new_tokens, temperature, steering, max_length
            if "The story should follow a character who is feeling" in prompt:
                return "[story 1]\nA short neutral-safe story.", 8
            return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.\n\nPerson: Next.\n\nAI: Done.", 12

        def model_hash(self) -> str:
            return "inline-backend-hash"

    # Fail fast if generation tries to build a new backend instead of using the injected one.
    monkeypatch.setattr("art.data.generate.create_backend", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected backend create")))

    rows = generate_probe_data(
        DataGenConfig(
            topics=["A test topic"],
            emotions=["afraid", "amazed"],
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        ),
        backend=_InlineBackend(),
    )
    assert len(rows) > 0


def test_data_generate_generation_cache_reuses_outputs(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _CountingBackend:
        backend_name = "mock"
        model_id = "mock/model"
        tokenizer_id = "mock/tokenizer"

        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            steering=None,
            max_length: int = 2048,
        ) -> tuple[str, int]:
            del max_new_tokens, temperature, steering, max_length
            self.calls += 1
            if "The story should follow a character who is feeling" in prompt:
                return "[story 1]\nA short neutral-safe story.", 8
            return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.\n\nPerson: Next.\n\nAI: Done.", 12

        def model_hash(self) -> str:
            return "cache-backend-hash"

    cfg = DataGenConfig(
        topics=["A test topic"],
        emotions=["afraid", "amazed"],
        stories_per_topic_emotion=1,
        dialogues_per_topic_emotion=0,
        neutral_dialogues_per_topic=1,
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        use_generation_cache=True,
        generation_cache_dir="cache/generation-test",
    )

    first_backend = _CountingBackend()
    rows_first = generate_probe_data(cfg, backend=first_backend)
    assert first_backend.calls > 0
    assert len(rows_first) > 0

    second_backend = _CountingBackend()
    rows_second = generate_probe_data(cfg, backend=second_backend)
    assert len(rows_second) == len(rows_first)
    assert second_backend.calls == 0


def test_data_generate_uses_batch_generation_when_available(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _BatchBackend:
        backend_name = "mock"
        model_id = "mock/model"
        tokenizer_id = "mock/tokenizer"

        def __init__(self) -> None:
            self.single_calls = 0
            self.batch_calls = 0

        def _respond(self, prompt: str) -> tuple[str, int]:
            if "The story should follow a character who is feeling" in prompt:
                return "[story 1]\nA short neutral-safe story.", 8
            return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.\n\nPerson: Next.\n\nAI: Done.", 12

        def generate(
            self,
            prompt: str,
            *,
            max_new_tokens: int,
            temperature: float,
            steering=None,
            max_length: int = 2048,
        ) -> tuple[str, int]:
            del max_new_tokens, temperature, steering, max_length
            self.single_calls += 1
            return self._respond(prompt)

        def generate_batch(
            self,
            prompts: list[str],
            *,
            max_new_tokens: int,
            temperature: float,
            steering=None,
            max_length: int = 2048,
        ) -> list[tuple[str, int]]:
            del max_new_tokens, temperature, steering, max_length
            self.batch_calls += 1
            return [self._respond(prompt) for prompt in prompts]

        def model_hash(self) -> str:
            return "batch-backend-hash"

    backend = _BatchBackend()
    rows = generate_probe_data(
        DataGenConfig(
            topics=["A test topic"],
            emotions=["afraid", "amazed"],
            stories_per_topic_emotion=1,
            dialogues_per_topic_emotion=0,
            neutral_dialogues_per_topic=2,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
            use_generation_cache=False,
            generation_batch_size=4,
        ),
        backend=backend,
    )
    assert len(rows) > 0
    assert backend.batch_calls >= 1
    assert backend.single_calls == 0
