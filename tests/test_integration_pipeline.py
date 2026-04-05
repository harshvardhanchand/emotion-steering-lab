from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from art.cli import app


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def test_integration_pipeline_mock(monkeypatch, tmp_path: Path) -> None:
    shutil.copytree(Path.cwd() / "schemas", tmp_path / "schemas")
    monkeypatch.setenv("ART_PROJECT_ROOT", str(tmp_path))

    runner = CliRunner()

    data_out = tmp_path / "data/probe_data.jsonl"
    r1 = runner.invoke(
        app,
        [
            "data",
            "generate",
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--topic",
            "Integration topic",
            "--emotion",
            "happy",
            "--emotion",
            "sad",
            "--out",
            str(data_out),
        ],
    )
    assert r1.exit_code == 0, r1.output
    assert data_out.exists()
    assert (tmp_path / "data/probe_data.jsonl.meta.json").exists()

    artifact_out = tmp_path / "artifacts/probe_artifact.json"
    r2 = runner.invoke(
        app,
        [
            "probe",
            "train",
            "--probe-data",
            str(data_out),
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--num-layers",
            "8",
            "--hidden-size",
            "32",
            "--out",
            str(artifact_out),
        ],
    )
    assert r2.exit_code == 0, r2.output
    assert artifact_out.exists()
    assert (tmp_path / "artifacts/probe_artifact.json.meta.json").exists()

    artifact = json.loads(artifact_out.read_text(encoding="utf-8"))
    probe_name = str(artifact["probes"][0]["probe_name"])

    diagnosis_out = tmp_path / "reports/diagnosis.jsonl"
    r3 = runner.invoke(
        app,
        [
            "probe",
            "diagnose",
            "--probe-artifact",
            str(artifact_out),
            "--cases",
            str(data_out),
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--out",
            str(diagnosis_out),
        ],
    )
    assert r3.exit_code == 0, r3.output
    assert diagnosis_out.exists()
    assert (tmp_path / "reports/diagnosis.jsonl.meta.json").exists()
    assert any(r.get("record_type") == "comparison_summary" for r in _jsonl_rows(diagnosis_out))

    sweep_out = tmp_path / "runs/steering_results.jsonl"
    sweep_md = tmp_path / "runs/steering_summary.md"
    sweep_html = tmp_path / "runs/steering_summary.html"
    r4 = runner.invoke(
        app,
        [
            "steer",
            "sweep",
            "--probe-artifact",
            str(artifact_out),
            "--probe-name",
            probe_name,
            "--cases",
            str(data_out),
            "--backend",
            "mock",
            "--model-id",
            "mock/model",
            "--tokenizer-id",
            "mock/tokenizer",
            "--alpha-min",
            "-0.05",
            "--alpha-max",
            "0.05",
            "--alpha-step",
            "0.05",
            "--out",
            str(sweep_out),
            "--report-md",
            str(sweep_md),
            "--report-html",
            str(sweep_html),
        ],
    )
    assert r4.exit_code == 0, r4.output
    assert sweep_out.exists()
    assert sweep_md.exists()
    assert sweep_html.exists()
    assert (tmp_path / "runs/steering_results.jsonl.meta.json").exists()
    assert _jsonl_rows(sweep_out)
    assert sweep_md.read_text(encoding="utf-8").strip()
    assert sweep_html.read_text(encoding="utf-8").strip()

