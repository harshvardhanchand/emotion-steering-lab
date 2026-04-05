from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from art.constants import EMOTION_WORDS, TOPICS
from art.data.generate import DataGenConfig, generate_probe_data
from art.errors import ArtError
import art.probes.train as train_module
from art.probes.train import train_probe_artifact
from art.schemas.validator import validate_document, validate_documents


def test_probe_train_artifact_schema(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:2],
            emotions=EMOTION_WORDS[:4],
            seed=11,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )
    validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")

    artifact = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        num_layers=12,
        hidden_size=64,
        backend_name="mock",
    )
    validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")
    assert len(artifact["probes"]) >= 1


def test_probe_train_requires_neutral_dialogues(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:1],
            emotions=EMOTION_WORDS[:3],
            seed=7,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )
    non_neutral_rows = [r for r in rows if str(r.get("source_type")) != "neutral_dialogue"]

    with pytest.raises(ArtError, match="neutral_dialogue"):
        train_probe_artifact(
            records=non_neutral_rows,
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
            backend_name="mock",
            num_layers=8,
            hidden_size=32,
        )


def test_probe_train_requires_at_least_two_emotions(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:1],
            emotions=EMOTION_WORDS[:2],
            seed=13,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )
    single_emotion_rows: list[dict[str, object]] = []
    keep_emotion = EMOTION_WORDS[0]
    for row in rows:
        source_type = str(row.get("source_type"))
        emotion = str(row.get("emotion_label"))
        if source_type == "story" and emotion != keep_emotion:
            continue
        if source_type == "dialogue" and emotion != keep_emotion:
            continue
        single_emotion_rows.append(row)

    with pytest.raises(ArtError, match="at least two"):
        train_probe_artifact(
            records=single_emotion_rows,  # type: ignore[arg-type]
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
            backend_name="mock",
            num_layers=8,
            hidden_size=32,
        )


def test_probe_train_skips_emotional_dialogue_extraction(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:1],
            emotions=EMOTION_WORDS[:2],
            seed=17,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )

    expected_extractions = sum(
        1
        for row in rows
        if str(row.get("source_type")) in {"story", "neutral_dialogue"}
    )

    class _CountingBackend:
        def __init__(self) -> None:
            self.backend_name = "mock"
            self.num_layers = 8
            self.hidden_size = 32
            self.model_id = "mock/model"
            self.tokenizer_id = "mock/tokenizer"
            self.calls = 0
            self.batch_calls = 0

        def extract_hidden_states(self, text: str, *, max_length: int):  # noqa: ARG002
            import numpy as np

            self.calls += 1
            return np.zeros((self.num_layers, 64, self.hidden_size), dtype=np.float32)

        def extract_hidden_states_batch(self, texts: list[str], *, max_length: int):  # noqa: ARG002
            import numpy as np

            self.batch_calls += 1
            return [np.zeros((self.num_layers, 64, self.hidden_size), dtype=np.float32) for _ in texts]

        def model_hash(self) -> str:
            return "counting-backend-hash"

    backend = _CountingBackend()
    monkeypatch.setattr(train_module, "create_backend", lambda **kwargs: backend)

    artifact = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        backend_name="mock",
        num_layers=8,
        hidden_size=32,
    )

    assert len(artifact["probes"]) >= 1
    assert backend.calls == 0
    assert backend.batch_calls >= 1
    assert expected_extractions >= 1


def test_probe_train_uses_injected_backend(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:1],
            emotions=EMOTION_WORDS[:2],
            seed=23,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )

    class _InlineBackend:
        def __init__(self) -> None:
            self.backend_name = "mock"
            self.num_layers = 8
            self.hidden_size = 32
            self.model_id = "mock/model"
            self.tokenizer_id = "mock/tokenizer"

        def extract_hidden_states(self, text: str, *, max_length: int):  # noqa: ARG002
            import numpy as np

            return np.zeros((self.num_layers, 64, self.hidden_size), dtype=np.float32)

        def extract_hidden_states_batch(self, texts: list[str], *, max_length: int):  # noqa: ARG002
            import numpy as np

            return [np.zeros((self.num_layers, 64, self.hidden_size), dtype=np.float32) for _ in texts]

        def model_hash(self) -> str:
            return "inline-backend-hash"

    monkeypatch.setattr(train_module, "create_backend", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("unexpected backend create")))

    artifact = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        backend_name="mock",
        num_layers=8,
        hidden_size=32,
        backend=_InlineBackend(),
    )
    assert len(artifact["probes"]) >= 1


def test_probe_train_activation_cache_reuses_extraction(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:1],
            emotions=EMOTION_WORDS[:2],
            seed=29,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )

    class _CountingBackend:
        def __init__(self) -> None:
            self.backend_name = "mock"
            self.num_layers = 8
            self.hidden_size = 32
            self.model_id = "mock/model"
            self.tokenizer_id = "mock/tokenizer"
            self.calls = 0

        def extract_hidden_states(self, text: str, *, max_length: int):  # noqa: ARG002
            import numpy as np

            self.calls += 1
            return np.zeros((self.num_layers, 64, self.hidden_size), dtype=np.float32)

        def model_hash(self) -> str:
            return "fixed-model-hash"

    cache_dir = tmp_path / "cache" / "activations"

    backend_first = _CountingBackend()
    artifact_first = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        backend_name="mock",
        num_layers=8,
        hidden_size=32,
        backend=backend_first,
        use_activation_cache=True,
        activation_cache_dir=cache_dir,
    )
    assert len(artifact_first["probes"]) >= 1
    assert backend_first.calls > 0

    backend_second = _CountingBackend()
    artifact_second = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        backend_name="mock",
        num_layers=8,
        hidden_size=32,
        backend=backend_second,
        use_activation_cache=True,
        activation_cache_dir=cache_dir,
    )
    assert len(artifact_second["probes"]) >= 1
    assert backend_second.calls == 0
