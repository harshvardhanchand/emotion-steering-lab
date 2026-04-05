from __future__ import annotations

import shutil
import time
from pathlib import Path

from art.artifacts.read import read_json
from art.artifacts.write import write_jsonl
from art.constants import EMOTION_WORDS, TOPICS
from art.data.generate import DataGenConfig, generate_probe_data
from art.ui_app import _new_build_job, _start_generate_train_job


def _wait_for_job(job: dict[str, object], timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        thread = job.get("thread")
        if thread is not None and not thread.is_alive():
            return
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for UI build job thread")


def test_generate_train_resumes_from_existing_probe_data(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    out_data = tmp_path / "runs" / "resume_case" / "probe_data.jsonl"
    out_artifact = tmp_path / "runs" / "resume_case" / "probe_artifact.json"
    out_data.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:1],
            emotions=EMOTION_WORDS[:2],
            stories_per_topic_emotion=1,
            dialogues_per_topic_emotion=0,
            neutral_dialogues_per_topic=2,
            seed=31,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
            max_new_tokens=128,
        )
    )
    write_jsonl(out_data, rows)

    job = _new_build_job("generate_train")
    cfg = DataGenConfig(
        topics=TOPICS[:1],
        emotions=EMOTION_WORDS[:2],
        stories_per_topic_emotion=1,
        dialogues_per_topic_emotion=0,
        neutral_dialogues_per_topic=2,
        seed=31,
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        max_new_tokens=128,
    )
    _start_generate_train_job(
        job=job,
        cfg=cfg,
        out_data=out_data,
        out_artifact=out_artifact,
        train_num_layers=8,
        train_hidden_size=32,
    )
    _wait_for_job(job)

    assert str(job.get("status")) == "completed"
    result = job.get("result")
    assert isinstance(result, dict)
    assert int(result.get("rows_count", 0)) == len(rows)
    assert out_artifact.exists()

    checkpoint = out_artifact.parent / f"{out_artifact.stem}.checkpoint.json"
    assert checkpoint.exists()
    payload = read_json(checkpoint)
    assert bool(payload.get("resumed_from_existing_data")) is True
    assert str(payload.get("status")) == "completed"

