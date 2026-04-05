from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from art.constants import SCHEMA_VERSION  # noqa: E402
from art.constants import EMOTION_WORDS, TOPICS  # noqa: E402
from art.data.generate import DataGenConfig, generate_probe_data  # noqa: E402
from art.probes.diagnose import run_diagnosis  # noqa: E402
from art.probes.train import train_probe_artifact  # noqa: E402
from art.schemas.validator import validate_document, validate_documents  # noqa: E402
from art.steering.profile import build_profile  # noqa: E402
from art.steering.run import run_steering  # noqa: E402
from art.steering.sweep import run_alpha_sweep  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_schema_files_parse() -> None:
    for path in sorted((ROOT / "schemas").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        _assert(isinstance(payload, dict), f"Schema must be JSON object: {path}")


def check_data_contract() -> list[dict[str, object]]:
    rows = generate_probe_data(
        DataGenConfig(
            topics=TOPICS[:2],
            emotions=EMOTION_WORDS[:3],
            seed=7,
            backend_name="mock",
            model_id="mock/model",
            tokenizer_id="mock/tokenizer",
        )
    )
    _assert(len(rows) > 0, "Generated probe data cannot be empty")
    validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")
    return rows


def check_probe_artifact_contract(rows: list[dict[str, object]]) -> dict[str, object]:
    artifact = train_probe_artifact(
        records=rows,
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        num_layers=16,
        hidden_size=64,
        backend_name="mock",
    )
    validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")
    _assert(len(list(artifact.get("probes", []))) > 0, "Artifact must contain probes")
    return artifact


def check_steering_contract(rows: list[dict[str, object]], artifact: dict[str, object]) -> None:
    probe_name = str(artifact["probes"][0]["probe_name"])
    results = run_steering(
        probe_artifact=artifact,
        probe_name=probe_name,
        alpha=0.05,
        cases=rows[:10],
        base_run_id="baseline",
        steer_run_id="steer_001",
        scope="full_suite",
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
    )
    validate_documents(results, "steering_results.schema.json", context_prefix="steering")

    profile = build_profile(
        profile_id="profile_001",
        model_id=str(artifact["model"]["model_id"]),
        probe_name=probe_name,
        layers=[int(artifact["layer_selection"]["selected_layer"])],
        alpha=0.05,
        scope="full_suite",
    )
    validate_document(profile, "steering_profile.schema.json", context="steering_profile")


def check_diagnosis_contract(rows: list[dict[str, object]], artifact: dict[str, object]) -> None:
    diagnosis = run_diagnosis(
        probe_artifact=artifact,
        cases=rows[:8],
        run_id="diag_001",
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
    )
    validate_documents(diagnosis, "diagnosis.schema.json", context_prefix="diagnosis")


def check_sweep_contract(rows: list[dict[str, object]], artifact: dict[str, object]) -> None:
    probe_name = str(artifact["probes"][0]["probe_name"])
    sweep_rows, md, html = run_alpha_sweep(
        probe_artifact=artifact,
        probe_name=probe_name,
        cases=rows[:10],
        base_run_id="baseline",
        sweep_run_id="sweep_001",
        scope="full_suite",
        alpha_min=-0.05,
        alpha_max=0.05,
        alpha_step=0.05,
        backend_name="mock",
        model_id="mock/model",
        tokenizer_id="mock/tokenizer",
        device="auto",
        dtype="auto",
        max_length=1024,
        max_new_tokens=48,
        temperature=0.0,
    )
    validate_documents(sweep_rows, "steering_results.schema.json", context_prefix="sweep")
    _assert(bool(md.strip()), "Sweep markdown report cannot be empty")
    _assert(bool(html.strip()), "Sweep HTML report cannot be empty")


def check_schema_version_constant() -> None:
    _assert(SCHEMA_VERSION == "1.0", "schema version must stay pinned at 1.0")


def check_run_id_format() -> None:
    from art.artifacts.paths import make_run_id  # noqa: E402

    run_id = make_run_id()
    _assert(
        bool(re.match(r"^\d{8}_\d{6}_[a-f0-9]{8}$", run_id)),
        f"run_id does not match contract: {run_id}",
    )


def main() -> None:
    check_schema_files_parse()
    check_schema_version_constant()
    check_run_id_format()
    rows = check_data_contract()
    artifact = check_probe_artifact_contract(rows)
    check_steering_contract(rows, artifact)
    check_diagnosis_contract(rows, artifact)
    check_sweep_contract(rows, artifact)
    print("Contract checks passed.")


if __name__ == "__main__":
    main()
