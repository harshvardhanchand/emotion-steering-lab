from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from art.cli import app
from art.constants import TOPICS
from art.data.generate import (
    DataGenConfig,
    _check_neutral_dialogue_qc,
    _to_human_assistant,
    generate_probe_data,
)
from art.errors import ArtError
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


def test_to_human_assistant_normalizes_common_speaker_formats() -> None:
    raw = """
1) User : Can you summarize this?
2) Assistant - Yes, share the text.

Speaker 1: Please keep it brief.
Speaker 2 — Sure.
"""
    normalized = _to_human_assistant(raw)
    assert "Human: Can you summarize this?" in normalized
    assert "Assistant: Yes, share the text." in normalized
    assert "Human: Please keep it brief." in normalized
    assert "Assistant: Sure." in normalized


def test_neutral_qc_allows_ambiguous_non_emotional_terms() -> None:
    text = "Human: Create a table of contents.\nAssistant: Here is a content outline with a kind structure."
    issues = _check_neutral_dialogue_qc(text)
    assert not any("neutral leakage terms found" in issue for issue in issues)


def test_neutral_qc_still_flags_clear_emotion_terms() -> None:
    text = "Human: I feel sad today.\nAssistant: Let's proceed with the task."
    issues = _check_neutral_dialogue_qc(text)
    assert any("neutral leakage terms found" in issue for issue in issues)


def test_data_generate_skips_failed_qc_rows_and_logs_drops(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _DropOneNeutralBackend:
        backend_name = "mock"
        model_id = "mock/model"
        tokenizer_id = "mock/tokenizer"

        def __init__(self) -> None:
            self.neutral_calls = 0

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
            if "CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless." in prompt:
                self.neutral_calls += 1
                if self.neutral_calls == 1 or "Regenerate from scratch." in prompt:
                    return "[dialogue 1]\nMissing speaker labels.", 8
                return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.", 10
            if "The story should follow a character who is feeling" in prompt:
                return "[story 1]\nThe character reacts through actions and body language.", 12
            return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.", 10

        def model_hash(self) -> str:
            return "a" * 64

    cfg = DataGenConfig(
        topics=["A test topic"],
        emotions=["afraid", "amazed"],
        stories_per_topic_emotion=1,
        dialogues_per_topic_emotion=0,
        neutral_dialogues_per_topic=3,
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        use_generation_cache=False,
        max_regen_attempts=2,
        max_drop_pct=0.25,
        max_drop_pct_per_source_type=1.0,
        max_drop_pct_per_emotion=1.0,
        min_required_neutral_rows=2,
        qc_drop_log_path="runs/test_qc_drops.jsonl",
        qc_drop_summary_path="runs/test_qc_drop_summary.jsonl",
    )

    rows = generate_probe_data(cfg, backend=_DropOneNeutralBackend())
    validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")
    assert sum(1 for r in rows if str(r.get("source_type")) == "story") == 2
    assert sum(1 for r in rows if str(r.get("source_type")) == "neutral_dialogue") == 2

    drop_log = tmp_path / "runs/test_qc_drops.jsonl"
    assert drop_log.exists()
    dropped = [json.loads(line) for line in drop_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(dropped) == 1
    assert str(dropped[0].get("record_id", "")).startswith("neutral_")
    assert dropped[0].get("issues")

    summary_log = tmp_path / "runs/test_qc_drop_summary.jsonl"
    assert summary_log.exists()
    summaries = [json.loads(line) for line in summary_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(summaries) == 1
    summary = summaries[0]
    assert int(summary.get("dropped_count", -1)) == 1
    assert int(summary.get("retained_items", -1)) == len(rows)
    assert isinstance(summary.get("dropped_by_reason"), dict)


def test_data_generate_errors_when_drop_rate_exceeds_threshold(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _AlwaysBadNeutralBackend:
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
            if "CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless." in prompt:
                return "[dialogue 1]\nMissing speaker labels.", 8
            if "The story should follow a character who is feeling" in prompt:
                return "[story 1]\nA compliant story.", 8
            return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.", 10

        def model_hash(self) -> str:
            return "b" * 64

    cfg = DataGenConfig(
        topics=["A test topic"],
        emotions=["afraid", "amazed"],
        stories_per_topic_emotion=1,
        dialogues_per_topic_emotion=0,
        neutral_dialogues_per_topic=2,
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        use_generation_cache=False,
        max_regen_attempts=2,
        max_drop_pct=0.10,
        min_required_neutral_rows=0,
        qc_drop_log_path="runs/test_qc_drops_exceeded.jsonl",
    )

    with pytest.raises(ArtError, match="exceeding max_drop_pct"):
        generate_probe_data(cfg, backend=_AlwaysBadNeutralBackend())


def test_data_generate_retry_bypasses_cache(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _RetryCacheBackend:
        backend_name = "mock"
        model_id = "mock/model"
        tokenizer_id = "mock/tokenizer"

        def __init__(self) -> None:
            self.neutral_calls = 0

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
            if "CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless." in prompt:
                self.neutral_calls += 1
                if self.neutral_calls == 1:
                    return "[dialogue 1]\nMissing speaker labels.", 8
                return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.", 10
            if "The story should follow a character who is feeling" in prompt:
                return "[story 1]\nA compliant story body.", 8
            return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.", 10

        def model_hash(self) -> str:
            return "c" * 64

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
        generation_cache_dir="cache/retry-cache-test",
        max_regen_attempts=2,
        max_drop_pct=1.0,
        max_drop_pct_per_source_type=1.0,
        max_drop_pct_per_emotion=1.0,
        min_required_neutral_rows=1,
    )

    backend = _RetryCacheBackend()
    rows = generate_probe_data(cfg, backend=backend)
    validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")
    assert backend.neutral_calls == 2
    assert sum(1 for r in rows if str(r.get("source_type")) == "neutral_dialogue") == 1


def test_data_generate_enforces_source_type_drop_cap(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _BadDialogueBackend:
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
            if "The Person should be feeling" in prompt:
                return "[dialogue 1]\nMissing speaker labels.", 8
            if "CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless." in prompt:
                return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.", 10
            return "[story 1]\nA compliant story body.", 8

        def model_hash(self) -> str:
            return "d" * 64

    cfg = DataGenConfig(
        topics=["A test topic"],
        emotions=["afraid", "amazed"],
        stories_per_topic_emotion=1,
        dialogues_per_topic_emotion=1,
        neutral_dialogues_per_topic=1,
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        use_generation_cache=False,
        max_regen_attempts=1,
        max_drop_pct=1.0,
        max_drop_pct_per_source_type=0.20,
        max_drop_pct_per_emotion=1.0,
        min_required_neutral_rows=1,
    )

    with pytest.raises(ArtError, match="source_type=dialogue"):
        generate_probe_data(cfg, backend=_BadDialogueBackend())


def test_data_generate_enforces_emotion_drop_cap(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _BadAfraidStoryBackend:
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
            if "The story should follow a character who is feeling afraid" in prompt:
                return "[story 1]\nThe person feels afraid.", 8
            if "The story should follow a character who is feeling" in prompt:
                return "[story 1]\nA compliant story body.", 8
            return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.", 10

        def model_hash(self) -> str:
            return "e" * 64

    cfg = DataGenConfig(
        topics=["A test topic"],
        emotions=["afraid", "amazed"],
        stories_per_topic_emotion=1,
        dialogues_per_topic_emotion=0,
        neutral_dialogues_per_topic=1,
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        use_generation_cache=False,
        max_regen_attempts=1,
        max_drop_pct=1.0,
        max_drop_pct_per_source_type=1.0,
        max_drop_pct_per_emotion=0.30,
        min_required_neutral_rows=1,
    )

    with pytest.raises(ArtError, match="emotion_label=afraid"):
        generate_probe_data(cfg, backend=_BadAfraidStoryBackend())


def test_data_generate_enforces_split_sanity_after_drops(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    class _DropSpecificValStoryBackend:
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
            if "Topic: T3" in prompt and "The story should follow a character who is feeling afraid" in prompt:
                return "[story 1]\nThe person feels afraid.", 8
            if "The story should follow a character who is feeling" in prompt:
                return "[story 1]\nA compliant story body.", 8
            return "[dialogue 1]\n\nPerson: Task.\n\nAI: Response.", 10

        def model_hash(self) -> str:
            return "f" * 64

    cfg = DataGenConfig(
        topics=["T1", "T2", "T3"],
        emotions=["afraid", "amazed"],
        stories_per_topic_emotion=1,
        dialogues_per_topic_emotion=0,
        neutral_dialogues_per_topic=2,
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        use_generation_cache=False,
        max_regen_attempts=1,
        max_drop_pct=1.0,
        max_drop_pct_per_source_type=1.0,
        max_drop_pct_per_emotion=1.0,
        min_required_split_count_per_class=1,
        min_required_neutral_rows=0,
    )

    with pytest.raises(ArtError, match="Post-drop split check failed"):
        generate_probe_data(cfg, backend=_DropSpecificValStoryBackend())


def test_data_heldout_cli_generates_test_split_without_topic_overlap(
    monkeypatch, tmp_path: Path
) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    runner = CliRunner()
    train_out = tmp_path / "data/train_probe_data.jsonl"
    r1 = runner.invoke(
        app,
        [
            "data",
            "generate",
            "--out",
            str(train_out),
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--topic",
            "Train topic only",
            "--emotion",
            "afraid",
            "--emotion",
            "amazed",
        ],
    )
    assert r1.exit_code == 0, r1.output

    heldout_out = tmp_path / "data/eval_cases.jsonl"
    r2 = runner.invoke(
        app,
        [
            "data",
            "heldout",
            "--train-probe-data",
            str(train_out),
            "--out",
            str(heldout_out),
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--topic",
            "Heldout topic only",
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert heldout_out.exists()
    assert (tmp_path / "data/eval_cases.jsonl.meta.json").exists()

    rows = [json.loads(line) for line in heldout_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) > 0
    validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")
    assert all(str(r.get("split")) == "test" for r in rows)
    assert all(str(r.get("topic")) == "Heldout topic only" for r in rows)
    assert {str(r.get("emotion_label")) for r in rows} == {"afraid", "amazed"}


def test_data_heldout_cli_rejects_explicit_topic_overlap(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    runner = CliRunner()
    train_out = tmp_path / "data/train_probe_data.jsonl"
    r1 = runner.invoke(
        app,
        [
            "data",
            "generate",
            "--out",
            str(train_out),
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--topic",
            TOPICS[0],
            "--emotion",
            "afraid",
            "--emotion",
            "amazed",
        ],
    )
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(
        app,
        [
            "data",
            "heldout",
            "--train-probe-data",
            str(train_out),
            "--out",
            str(tmp_path / "data/eval_cases.jsonl"),
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--topic",
            TOPICS[0],
        ],
    )
    assert r2.exit_code != 0
    assert "overlap" in r2.output.lower()
