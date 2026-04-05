from __future__ import annotations

import json
import shutil
from pathlib import Path

from art.constants import EMOTION_WORDS, TOPICS
from typer.testing import CliRunner

from art.cli import app
from art.data.generate import DataGenConfig, generate_probe_data
from art.probes.train import train_probe_artifact
from art.steering.run import run_steering


def test_steer_run_cli(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:2],
            emotions=EMOTION_WORDS[:3],
            seed=5,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )
    data_path = tmp_path / "data/probe_data.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    artifact = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        num_layers=12,
        hidden_size=64,
        backend_name="mock",
    )
    artifact_path = tmp_path / "artifacts/probe_artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    probe_name = artifact["probes"][0]["probe_name"]

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "steer",
            "run",
            "--probe-artifact",
            "artifacts/probe_artifact.json",
            "--probe-name",
            probe_name,
            "--cases",
            "data/probe_data.jsonl",
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--save-profile",
            "profiles/default.json",
        ],
    )
    assert result.exit_code == 0, result.output

    assert (tmp_path / "profiles/default.json").exists()
    run_dirs = sorted((tmp_path / "runs").glob("*"))
    assert run_dirs
    assert (run_dirs[-1] / "steering_results.jsonl").exists()
    assert (run_dirs[-1] / "steering_results.jsonl.meta.json").exists()


def test_failures_only_uses_actual_baseline_failures() -> None:
    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:2],
            emotions=EMOTION_WORDS[:3],
            seed=9,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )
    for idx, row in enumerate(rows):
        row["baseline_final_severity"] = 3 if idx % 4 == 0 else 0

    artifact = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        num_layers=10,
        hidden_size=48,
        backend_name="mock",
    )
    probe_name = artifact["probes"][0]["probe_name"]

    out = run_steering(
        probe_artifact=artifact,
        probe_name=probe_name,
        alpha=0.05,
        cases=rows,
        base_run_id="baseline",
        steer_run_id="steer_test",
        scope="failures_only",
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
    )
    sample_rows = [r for r in out if r["record_type"] == "sample"]
    assert sample_rows
    assert all(bool(r["baseline_failure"]) for r in sample_rows)

    deltas = {round(float(r["quality_delta"]), 6) for r in sample_rows}
    assert len(deltas) > 1


def test_steer_run_cli_applies_profile(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:2],
            emotions=EMOTION_WORDS[:3],
            seed=11,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )
    data_path = tmp_path / "data/probe_data.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    artifact = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        num_layers=12,
        hidden_size=64,
        backend_name="mock",
    )
    artifact_path = tmp_path / "artifacts/probe_artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    probe_name = str(artifact["probes"][0]["probe_name"])
    profile_path = tmp_path / "profiles/default.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "schema_version": "1.0",
        "profile_id": "profile_test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "model_id": "mock/model",
        "probe_name": probe_name,
        "layers": [int(artifact["layer_selection"]["selected_layer"])],
        "alpha": 0.03,
        "scope": "full_suite",
        "enabled": True,
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "steer",
            "run",
            "--probe-artifact",
            "artifacts/probe_artifact.json",
            "--profile",
            "profiles/default.json",
            "--cases",
            "data/probe_data.jsonl",
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
        ],
    )
    assert result.exit_code == 0, result.output


def test_steer_run_cli_applies_slider_profile(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:1],
            emotions=EMOTION_WORDS[:3],
            seed=13,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )
    data_path = tmp_path / "data/probe_data.jsonl"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    artifact = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        num_layers=12,
        hidden_size=64,
        backend_name="mock",
    )
    artifact_path = tmp_path / "artifacts/probe_artifact.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    probe_name = str(artifact["probes"][0]["probe_name"])
    profile_path = tmp_path / "profiles/slider.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    slider_profile = {
        "schema_version": "1.0",
        "profile_type": "slider_blend_v1",
        "profile_id": "blend_test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "model_id": "mock/model",
        "tokenizer_id": "mock/tokenizer",
        "selected_layer": int(artifact["layer_selection"]["selected_layer"]),
        "weights": {probe_name: 0.02},
        "enabled": True,
    }
    profile_path.write_text(json.dumps(slider_profile), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "steer",
            "run",
            "--probe-artifact",
            "artifacts/probe_artifact.json",
            "--profile",
            "profiles/slider.json",
            "--cases",
            "data/probe_data.jsonl",
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
        ],
    )
    assert result.exit_code == 0, result.output
