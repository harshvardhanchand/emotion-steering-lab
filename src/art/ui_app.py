"""Streamlit UI for one-click emotion steering studio."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from art.artifacts.read import read_json, read_jsonl
from art.artifacts.write import write_json, write_jsonl
from art.backends import SteeringIntervention, create_backend
from art.constants import DEFAULT_MAX_LENGTH, EMOTION_WORDS, PAPER_MODE_NAME, TOPICS, project_root
from art.data.generate import DataGenConfig, generate_probe_data, generation_config_hash
from art.errors import ArtError
from art.probes.train import train_probe_artifact
from art.repro import utc_now_iso
from art.schemas.validator import validate_document, validate_documents


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (project_root() / p)


def _default_emotions() -> list[str]:
    preferred = ["happy", "sad", "desperate"]
    out = [x for x in preferred if x in EMOTION_WORDS]
    return out if out else EMOTION_WORDS[:3]


def _inject_css(st) -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1200px; padding-top: 1.6rem; padding-bottom: 2rem;}
        .card {border: 1px solid #dde3ea; border-radius: 14px; padding: 1rem 1rem 0.7rem 1rem; background: #ffffff;}
        .pill {display:inline-block; padding:0.2rem 0.5rem; border-radius:999px; background:#eef3f8; border:1px solid #d5dde7; font-size:0.8rem; margin-right:0.4rem;}
        .hint {color:#4f5d6b; font-size:0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _probe_vector_map(artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(p["probe_name"]): np.asarray(p["vector"], dtype=np.float32)
        for p in artifact.get("probes", [])
    }


def _combined_weights_vector(artifact: dict[str, Any], weights: dict[str, float]) -> tuple[np.ndarray, int, float]:
    probe_map = _probe_vector_map(artifact)
    if not probe_map:
        raise ArtError("Artifact has no probes")

    dim = len(next(iter(probe_map.values())))
    combined = np.zeros(dim, dtype=np.float32)
    for name, alpha in weights.items():
        vec = probe_map.get(name)
        if vec is None:
            continue
        combined = combined + (float(alpha) * vec)

    magnitude = float(np.linalg.norm(combined))
    selected_layer = int(artifact["layer_selection"]["selected_layer"])
    return combined, selected_layer, magnitude


def _save_slider_profile(path: Path, *, artifact: dict[str, Any], weights: dict[str, float]) -> None:
    model = artifact.get("model", {})
    payload = {
        "schema_version": "1.0",
        "profile_type": "slider_blend_v1",
        "profile_id": f"blend_{utc_now_iso().replace(':', '').replace('-', '')}",
        "created_at": utc_now_iso(),
        "paper_mode": PAPER_MODE_NAME,
        "model_id": str(model.get("model_id", "")),
        "tokenizer_id": str(model.get("tokenizer_id", "")),
        "selected_layer": int(artifact["layer_selection"]["selected_layer"]),
        "weights": {k: float(v) for k, v in sorted(weights.items())},
        "enabled": True,
    }
    write_json(path, payload)


def _load_slider_profile_weights(path: Path, *, artifact: dict[str, Any]) -> dict[str, float]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ArtError(f"Invalid profile document: {path}")

    model = artifact.get("model", {})
    artifact_model_id = str(model.get("model_id", "")).strip()

    profile_type = str(payload.get("profile_type", "")).strip()
    if profile_type == "slider_blend_v1":
        profile_model_id = str(payload.get("model_id", "")).strip()
        if profile_model_id and artifact_model_id and profile_model_id != artifact_model_id:
            raise ArtError(
                f"Profile model_id ({profile_model_id}) does not match loaded artifact model_id ({artifact_model_id})"
            )
        weights_raw = payload.get("weights")
        if not isinstance(weights_raw, dict):
            raise ArtError("Invalid slider profile: missing `weights` object")
        probe_names = [str(p["probe_name"]) for p in artifact.get("probes", [])]
        if not probe_names:
            raise ArtError("Loaded artifact has no probes")
        return {name: float(weights_raw.get(name, 0.0)) for name in probe_names}

    validate_document(payload, "steering_profile.schema.json", context="steering_profile")
    profile_model_id = str(payload.get("model_id", "")).strip()
    if profile_model_id and artifact_model_id and profile_model_id != artifact_model_id:
        raise ArtError(
            f"Profile model_id ({profile_model_id}) does not match loaded artifact model_id ({artifact_model_id})"
        )
    probe_name = str(payload.get("probe_name", "")).strip()
    alpha = float(payload.get("alpha", 0.0))
    probe_names = [str(p["probe_name"]) for p in artifact.get("probes", [])]
    if probe_name not in probe_names:
        raise ArtError(f"Profile probe '{probe_name}' not found in loaded artifact")
    weights = {name: 0.0 for name in probe_names}
    weights[probe_name] = alpha
    return weights


def _sync_slider_state(st, probe_names: list[str]) -> None:
    state = st.session_state.get("slider_weights", {})
    if not isinstance(state, dict):
        state = {}
    next_state: dict[str, float] = {}
    for name in probe_names:
        next_state[name] = float(state.get(name, 0.0))
    st.session_state["slider_weights"] = next_state


def _current_weights(st, probe_names: list[str]) -> dict[str, float]:
    current: dict[str, float] = {}
    for name in probe_names:
        key = f"slider__{name}"
        current[name] = float(st.session_state.get(key, st.session_state.get("slider_weights", {}).get(name, 0.0)))
    st.session_state["slider_weights"] = current
    return current


def _default_run_outputs(st) -> tuple[str, str, str]:
    if "studio_run_tag" not in st.session_state:
        tag = utc_now_iso().replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        st.session_state["studio_run_tag"] = tag
    tag = str(st.session_state["studio_run_tag"])
    return (
        f"runs/ui_{tag}/probe_data.jsonl",
        f"runs/ui_{tag}/probe_artifact.json",
        f"profiles/slider_profile_{tag}.json",
    )


def _get_runtime_backend(st, *, backend_name: str, model_id: str, tokenizer_id: str, device: str, dtype: str, num_layers: int, hidden_size: int):
    key = (backend_name, model_id, tokenizer_id, device, dtype, int(num_layers), int(hidden_size))
    if st.session_state.get("runtime_backend_key") == key and st.session_state.get("runtime_backend") is not None:
        return st.session_state["runtime_backend"]

    backend = create_backend(
        backend_name=backend_name,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        device=device,
        dtype=dtype,
        num_layers=int(num_layers) if backend_name == "mock" else None,
        hidden_size=int(hidden_size) if backend_name == "mock" else None,
    )
    st.session_state["runtime_backend"] = backend
    st.session_state["runtime_backend_key"] = key
    return backend


def _new_build_job(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": "running",
        "progress": 0.0,
        "message": "Starting",
        "cancel_requested": False,
        "error": "",
        "result": None,
        "applied": False,
        "thread": None,
    }


def _set_job_progress(job: dict[str, Any], value: float, message: str) -> None:
    job["progress"] = min(1.0, max(0.0, float(value)))
    job["message"] = str(message)


def _checkpoint_path_for_artifact(out_artifact: Path) -> Path:
    return out_artifact.parent / f"{out_artifact.stem}.checkpoint.json"


def _write_build_checkpoint(
    path: Path,
    *,
    status: str,
    stage: str,
    out_data: Path,
    out_artifact: Path,
    message: str = "",
    rows_count: int | None = None,
    resumed_from_existing_data: bool = False,
    error: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "pipeline": "ui_generate_train_v1",
        "updated_at": utc_now_iso(),
        "status": status,
        "stage": stage,
        "message": message,
        "out_data": str(out_data),
        "out_artifact": str(out_artifact),
        "resumed_from_existing_data": bool(resumed_from_existing_data),
    }
    if rows_count is not None:
        payload["rows_count"] = int(rows_count)
    if error:
        payload["error"] = str(error)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _single_metadata_hash(rows: list[dict[str, Any]], field: str) -> str:
    values: set[str] = set()
    for idx, row in enumerate(rows, start=1):
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise ArtError(f"Cannot resume: probe_data[{idx}] missing metadata object")
        value = str(metadata.get(field, "")).strip()
        if not value:
            raise ArtError(f"Cannot resume: probe_data[{idx}] missing metadata.{field}")
        values.add(value)
    if len(values) != 1:
        raise ArtError(f"Cannot resume: probe_data contains multiple metadata.{field} values")
    return next(iter(values))


def _validate_resume_probe_data_integrity(*, rows: list[dict[str, Any]], cfg: DataGenConfig, model_hash: str) -> None:
    expected_cfg_hash = generation_config_hash(
        cfg,
        topics=list(cfg.topics) if cfg.topics else list(TOPICS),
        emotions=list(cfg.emotions) if cfg.emotions else list(EMOTION_WORDS),
    )
    actual_cfg_hash = _single_metadata_hash(rows, "generation_config_hash")
    if actual_cfg_hash != expected_cfg_hash:
        raise ArtError(
            "Cannot resume: probe_data generation_config_hash mismatch. "
            f"expected={expected_cfg_hash}, found={actual_cfg_hash}"
        )

    actual_model_hash = _single_metadata_hash(rows, "generation_model_hash")
    if actual_model_hash != model_hash:
        raise ArtError(
            "Cannot resume: probe_data generation_model_hash mismatch. "
            f"expected={model_hash}, found={actual_model_hash}"
        )


def _start_generate_train_job(
    *,
    job: dict[str, Any],
    cfg: DataGenConfig,
    out_data: Path,
    out_artifact: Path,
    train_num_layers: int,
    train_hidden_size: int,
) -> None:
    def _worker() -> None:
        checkpoint_path = _checkpoint_path_for_artifact(out_artifact)
        stage = "init"
        resumed = False
        try:
            if out_artifact.exists():
                raise ArtError(f"Output exists (immutable): {out_artifact}")
            _write_build_checkpoint(
                checkpoint_path,
                status="running",
                stage=stage,
                out_data=out_data,
                out_artifact=out_artifact,
                message="Starting",
            )

            should_cancel = lambda: bool(job.get("cancel_requested"))
            _set_job_progress(job, 0.01, "Loading backend")
            backend = create_backend(
                backend_name=cfg.backend_name,
                model_id=cfg.model_id,
                tokenizer_id=cfg.tokenizer_id or cfg.model_id,
                device=cfg.device,
                dtype=cfg.dtype,
                num_layers=int(train_num_layers) if cfg.backend_name == "mock" else None,
                hidden_size=int(train_hidden_size) if cfg.backend_name == "mock" else None,
            )
            backend_model_hash = backend.model_hash()
            rows: list[dict[str, Any]]
            if out_data.exists():
                resumed = True
                stage = "generation_complete"
                _set_job_progress(job, 0.52, "Found existing probe data. Validating for resume")
                rows = read_jsonl(out_data)
                validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")
                _validate_resume_probe_data_integrity(rows=rows, cfg=cfg, model_hash=backend_model_hash)
                _write_build_checkpoint(
                    checkpoint_path,
                    status="running",
                    stage=stage,
                    out_data=out_data,
                    out_artifact=out_artifact,
                    message="Resuming from existing probe data",
                    rows_count=len(rows),
                    resumed_from_existing_data=True,
                )
                _set_job_progress(job, 0.65, f"Resume ready ({len(rows)} rows)")
            else:
                stage = "generation"
                _write_build_checkpoint(
                    checkpoint_path,
                    status="running",
                    stage=stage,
                    out_data=out_data,
                    out_artifact=out_artifact,
                    message="Generation started",
                )
                _set_job_progress(job, 0.03, "Generation started")
                rows = generate_probe_data(
                    cfg,
                    progress_callback=lambda frac, msg: _set_job_progress(
                        job, 0.05 + (0.55 * float(frac)), f"Generating data: {msg}"
                    ),
                    should_cancel=should_cancel,
                    backend=backend,
                )
                validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")
                out_data.parent.mkdir(parents=True, exist_ok=True)
                write_jsonl(out_data, rows)
                stage = "generation_complete"
                _write_build_checkpoint(
                    checkpoint_path,
                    status="running",
                    stage=stage,
                    out_data=out_data,
                    out_artifact=out_artifact,
                    message="Probe data generated",
                    rows_count=len(rows),
                    resumed_from_existing_data=False,
                )
                _set_job_progress(job, 0.65, "Probe data validated and written")

            stage = "training"
            _write_build_checkpoint(
                checkpoint_path,
                status="running",
                stage=stage,
                out_data=out_data,
                out_artifact=out_artifact,
                message="Training probes",
                rows_count=len(rows),
                resumed_from_existing_data=resumed,
            )
            artifact = train_probe_artifact(
                records=rows,
                model_id=cfg.model_id,
                tokenizer_id=cfg.tokenizer_id or cfg.model_id,
                num_layers=int(train_num_layers),
                hidden_size=int(train_hidden_size),
                backend_name=cfg.backend_name,
                device=cfg.device,
                dtype=cfg.dtype,
                max_length=int(cfg.max_length),
                progress_callback=lambda frac, msg: _set_job_progress(
                    job, 0.66 + (0.30 * float(frac)), f"Training probes: {msg}"
                ),
                should_cancel=should_cancel,
                backend=backend,
            )
            validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")
            _set_job_progress(job, 0.97, "Probe artifact validated")

            out_artifact.parent.mkdir(parents=True, exist_ok=True)
            write_json(out_artifact, artifact)
            stage = "completed"
            _write_build_checkpoint(
                checkpoint_path,
                status="completed",
                stage=stage,
                out_data=out_data,
                out_artifact=out_artifact,
                message="Generate + Train complete",
                rows_count=len(rows),
                resumed_from_existing_data=resumed,
            )
            _set_job_progress(job, 1.0, "Generate + Train complete")

            job["result"] = {
                "rows_count": len(rows),
                "probe_count": len(artifact.get("probes", [])),
                "data_path": str(out_data),
                "artifact_path": str(out_artifact),
                "artifact": artifact,
                "chat_backend_cfg": {
                    "backend_name": cfg.backend_name,
                    "model_id": cfg.model_id,
                    "tokenizer_id": cfg.tokenizer_id or cfg.model_id,
                    "device": cfg.device,
                    "dtype": cfg.dtype,
                    "max_length": int(cfg.max_length),
                    "max_new_tokens": 256,
                    "temperature": 0.0,
                    "num_layers": int(artifact["model"]["num_layers"]),
                    "hidden_size": int(artifact["model"]["hidden_size"]),
                },
            }
            job["status"] = "completed"
        except Exception as exc:
            text = str(exc)
            if job.get("cancel_requested") or "cancelled by user" in text.lower():
                job["status"] = "cancelled"
                job["message"] = "Run cancelled by user"
                _write_build_checkpoint(
                    checkpoint_path,
                    status="cancelled",
                    stage=stage,
                    out_data=out_data,
                    out_artifact=out_artifact,
                    message="Run cancelled by user",
                    resumed_from_existing_data=resumed,
                    error=text,
                )
            else:
                job["status"] = "error"
                job["error"] = text
                _write_build_checkpoint(
                    checkpoint_path,
                    status="error",
                    stage=stage,
                    out_data=out_data,
                    out_artifact=out_artifact,
                    message="Run failed",
                    resumed_from_existing_data=resumed,
                    error=text,
                )

    thread = threading.Thread(target=_worker, daemon=True)
    job["thread"] = thread
    thread.start()


def _start_train_only_job(
    *,
    job: dict[str, Any],
    probe_data_path: Path,
    out_artifact: Path,
    backend_name: str,
    model_id: str,
    tokenizer_id: str,
    device: str,
    dtype: str,
    max_length: int,
    train_num_layers: int,
    train_hidden_size: int,
) -> None:
    def _worker() -> None:
        try:
            if out_artifact.exists():
                raise ArtError(f"Output exists (immutable): {out_artifact}")
            if not probe_data_path.exists():
                raise ArtError(f"Probe data not found: {probe_data_path}")

            should_cancel = lambda: bool(job.get("cancel_requested"))
            _set_job_progress(job, 0.05, "Loading existing probe data")
            rows = read_jsonl(probe_data_path)
            validate_documents(rows, "probe_data.schema.json", context_prefix="probe_data")
            _set_job_progress(job, 0.20, f"Loaded probe data ({len(rows)} rows)")

            artifact = train_probe_artifact(
                records=rows,
                model_id=model_id,
                tokenizer_id=tokenizer_id or model_id,
                num_layers=int(train_num_layers),
                hidden_size=int(train_hidden_size),
                backend_name=backend_name,
                device=device,
                dtype=dtype,
                max_length=int(max_length),
                progress_callback=lambda frac, msg: _set_job_progress(
                    job, 0.20 + (0.75 * float(frac)), f"Training probes: {msg}"
                ),
                should_cancel=should_cancel,
            )
            validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")
            out_artifact.parent.mkdir(parents=True, exist_ok=True)
            write_json(out_artifact, artifact)
            _set_job_progress(job, 1.0, "Train from existing data complete")

            job["result"] = {
                "rows_count": len(rows),
                "probe_count": len(artifact.get("probes", [])),
                "data_path": str(probe_data_path),
                "artifact_path": str(out_artifact),
                "artifact": artifact,
                "chat_backend_cfg": {
                    "backend_name": backend_name,
                    "model_id": model_id,
                    "tokenizer_id": tokenizer_id or model_id,
                    "device": device,
                    "dtype": dtype,
                    "max_length": int(max_length),
                    "max_new_tokens": 256,
                    "temperature": 0.0,
                    "num_layers": int(artifact["model"]["num_layers"]),
                    "hidden_size": int(artifact["model"]["hidden_size"]),
                },
            }
            job["status"] = "completed"
        except Exception as exc:
            text = str(exc)
            if job.get("cancel_requested") or "cancelled by user" in text.lower():
                job["status"] = "cancelled"
                job["message"] = "Run cancelled by user"
            else:
                job["status"] = "error"
                job["error"] = text

    thread = threading.Thread(target=_worker, daemon=True)
    job["thread"] = thread
    thread.start()


def _apply_job_result(st) -> None:
    job = st.session_state.get("build_job")
    if not isinstance(job, dict):
        return
    if job.get("status") != "completed" or job.get("applied"):
        return
    result = job.get("result") or {}
    artifact = result.get("artifact")
    if not isinstance(artifact, dict):
        return

    probe_names = [str(p["probe_name"]) for p in artifact.get("probes", [])]
    st.session_state["probe_data_path"] = str(result.get("data_path", ""))
    st.session_state["artifact_path"] = str(result.get("artifact_path", ""))
    st.session_state["active_artifact"] = artifact
    st.session_state["slider_weights"] = {name: 0.0 for name in probe_names}
    chat_cfg = result.get("chat_backend_cfg")
    if isinstance(chat_cfg, dict):
        st.session_state["chat_backend_cfg"] = chat_cfg
    job["applied"] = True


def _candidate_paths(*, suffix: str, must_contain: tuple[str, ...], limit: int = 120) -> list[str]:
    root = project_root()
    out: list[str] = []
    for base in (root / "runs", root / "data", root / "artifacts", root / "profiles"):
        if not base.exists() or not base.is_dir():
            continue
        for p in base.rglob(f"*{suffix}"):
            path_str = str(p)
            if must_contain and not any(token in path_str for token in must_contain):
                continue
            out.append(path_str)
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break
    out.sort(reverse=True)
    return out


def _estimate_generation_items(*, n_topics: int, n_emotions: int, stories_per_topic_emotion: int, dialogues_per_topic_emotion: int, neutral_dialogues_per_topic: int) -> int:
    per_topic = (n_emotions * (stories_per_topic_emotion + dialogues_per_topic_emotion)) + neutral_dialogues_per_topic
    return max(0, n_topics * per_topic)


def _friendly_error(raw: str) -> str:
    text = (raw or "").strip()
    low = text.lower()
    if "output exists (immutable)" in low:
        return f"{text}\n\nUse a new output path for this run."
    if "cancelled by user" in low:
        return "Run cancelled by user."
    if "out of memory" in low or "mps backend out of memory" in low or "cuda out of memory" in low:
        return f"{text}\n\nTry a smaller model, reduce max length, or reduce max new tokens."
    if "requires `transformers` and `torch`" in low:
        return f"{text}\n\nInstall dependencies with: `uv sync --extra hf`."
    if "not found" in low and "probe data" in low:
        return f"{text}\n\nPick an existing `probe_data.jsonl` file or generate one first."
    if "at least two non-neutral emotion labels" in low:
        return f"{text}\n\nUse at least 2 emotions for probe training."
    if "needs neutral_dialogue records" in low:
        return f"{text}\n\nUse paper-faithful generated data that includes neutral dialogues."
    return text or "Unknown error."


def _projection_score(
    *,
    backend: Any,
    prompt: str,
    response: str,
    response_token_count: int,
    vector: np.ndarray,
    layer: int,
    max_length: int,
) -> float:
    convo = f"Human: {prompt}\nAssistant: {response}"
    hidden = backend.extract_hidden_states(convo, max_length=max_length)
    if layer < 0 or layer >= hidden.shape[0]:
        raise ArtError(f"Selected layer {layer} is out of range")
    seq_len = int(hidden.shape[1])
    response_tokens = max(1, min(response_token_count, seq_len))
    start = max(0, seq_len - response_tokens)
    first_n = min(20, seq_len - start)
    end = start + max(1, first_n)
    pooled = hidden[layer, start:end, :].mean(axis=0)
    return float(np.dot(pooled.astype(np.float32), vector.astype(np.float32)))


def _severity_bucket(score: float) -> int:
    x = abs(float(score))
    if x >= 0.35:
        return 3
    if x >= 0.12:
        return 2
    return 1


def _job_is_running(job: dict[str, Any] | None) -> bool:
    if not isinstance(job, dict):
        return False
    if str(job.get("status")) != "running":
        return False
    thread = job.get("thread")
    return bool(thread is not None and getattr(thread, "is_alive", lambda: False)())


def render() -> None:
    import streamlit as st

    st.set_page_config(page_title="Steering Studio", layout="wide")
    _inject_css(st)

    st.title("Emotion Steering Studio")
    st.caption("Select emotions -> generate + train -> steer with sliders in live chat.")
    st.markdown(f"<span class='pill'>paper_mode={PAPER_MODE_NAME}</span>", unsafe_allow_html=True)

    if "build_job" not in st.session_state:
        st.session_state["build_job"] = None
    if "build_history" not in st.session_state:
        st.session_state["build_history"] = []
    if "studio_chat_history" not in st.session_state:
        st.session_state["studio_chat_history"] = []
    if "slider_weights" not in st.session_state:
        st.session_state["slider_weights"] = {}
    if "build_mode" not in st.session_state:
        st.session_state["build_mode"] = "generate_train"
    default_data_path, default_artifact_path, default_profile_path = _default_run_outputs(st)
    _apply_job_result(st)

    job = st.session_state.get("build_job")
    if isinstance(job, dict) and str(job.get("status")) == "running" and not _job_is_running(job):
        if bool(job.get("cancel_requested")):
            job["status"] = "cancelled"
            job["message"] = "Run cancelled by user"
        else:
            job["status"] = "error"
            job["error"] = "Background run stopped unexpectedly. Try again."

    if isinstance(job, dict) and str(job.get("status")) in {"completed", "cancelled", "error"} and not bool(job.get("history_logged")):
        result = job.get("result") or {}
        st.session_state["build_history"].insert(
            0,
            {
                "at": utc_now_iso(),
                "mode": str(job.get("mode", "")),
                "status": str(job.get("status", "")),
                "message": str(job.get("message", "")),
                "error": str(job.get("error", "")),
                "rows_count": int(result.get("rows_count", 0)),
                "probe_count": int(result.get("probe_count", 0)),
                "data_path": str(result.get("data_path", "")),
                "artifact_path": str(result.get("artifact_path", "")),
            },
        )
        job["history_logged"] = True

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("1) Build Emotion Vectors")
    st.markdown(
        "<p class='hint'>Use one mode per run: create new data+vectors, train from existing data, or load an existing artifact.</p>",
        unsafe_allow_html=True,
    )

    mode_options = [
        "generate_train",
        "train_existing_data",
        "load_existing_artifact",
    ]
    mode_label = {
        "generate_train": "Generate + Train (new run)",
        "train_existing_data": "Train from Existing Data",
        "load_existing_artifact": "Load Existing Artifact",
    }
    build_mode = st.selectbox(
        "Build mode",
        options=mode_options,
        format_func=lambda m: mode_label.get(m, m),
        key="build_mode",
    )

    p1, p2 = st.columns(2)
    with p1:
        gen_model_id = st.text_input("Model ID", value="Qwen/Qwen2.5-0.5B-Instruct", key="gen_model")
    with p2:
        out_artifact = st.text_input(
            "Probe artifact output path",
            value=st.session_state.get("artifact_path", default_artifact_path),
            disabled=(build_mode == "load_existing_artifact"),
        )

    emotions = st.multiselect(
        "Emotions",
        EMOTION_WORDS,
        default=_default_emotions(),
        disabled=(build_mode != "generate_train"),
    )

    # Advanced defaults (overridable via advanced controls).
    gen_backend = "transformers"
    gen_device = "auto"
    gen_dtype = "auto"
    seed = 42
    max_length = DEFAULT_MAX_LENGTH
    max_new_tokens = 768
    train_hidden_size = 256
    train_num_layers = 32
    use_full_topics = True
    generation_batch_size = 8
    generation_cache = True
    generation_cache_dir = "cache/generation"
    out_data = st.session_state.get("probe_data_path", default_data_path)
    selected_topics: list[str] = list(TOPICS)

    with st.expander("Advanced", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            gen_backend = st.selectbox(
                "Backend",
                options=["transformers", "transformerlens", "mock"],
                index=0,
                key="gen_backend",
            )
        with c2:
            gen_device = st.selectbox("Device", options=["auto", "cpu", "mps", "cuda"], index=0, key="gen_device")
        with c3:
            gen_dtype = st.selectbox("Dtype", options=["auto", "float32", "float16", "bfloat16"], index=0, key="gen_dtype")
        with c4:
            seed = st.number_input("Seed", min_value=0, max_value=100000, value=42)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            max_length = st.number_input("Max length", min_value=64, max_value=8192, value=DEFAULT_MAX_LENGTH)
        with c6:
            max_new_tokens = st.number_input("Max new tokens", min_value=64, max_value=4096, value=768)
        with c7:
            train_num_layers = st.number_input("Num layers (mock)", min_value=1, max_value=256, value=32)
        with c8:
            train_hidden_size = st.number_input("Hidden size (mock)", min_value=16, max_value=16384, value=256)

        c9, c10 = st.columns(2)
        with c9:
            generation_batch_size = st.number_input(
                "Generation batch size",
                min_value=1,
                max_value=256,
                value=8,
                disabled=(build_mode != "generate_train"),
            )
        with c10:
            generation_cache = st.checkbox(
                "Generation cache",
                value=True,
                disabled=(build_mode != "generate_train"),
            )
        generation_cache_dir = st.text_input(
            "Generation cache dir",
            value="cache/generation",
            disabled=(build_mode != "generate_train" or not generation_cache),
        )

        use_full_topics = st.checkbox(
            "Use full paper topic set",
            value=True,
            disabled=(build_mode != "generate_train"),
        )
        selected_topics = (
            list(TOPICS)
            if use_full_topics
            else st.multiselect(
                "Topic override",
                TOPICS,
                default=TOPICS[:2],
                disabled=(build_mode != "generate_train"),
            )
        )
        out_data = st.text_input(
            "Probe data output path",
            value=st.session_state.get("probe_data_path", default_data_path),
            disabled=(build_mode != "generate_train"),
        )

    if build_mode == "generate_train":
        st.caption(
            f"Using {len(selected_topics)} topics x {len(emotions)} emotions "
            f"(stories=1, emotional_dialogues=0, neutral_dialogues=2)."
        )

    existing_data_candidates = _candidate_paths(suffix=".jsonl", must_contain=("probe_data",))
    existing_artifact_candidates = _candidate_paths(suffix=".json", must_contain=("probe_artifact",))

    existing_probe_data_path = st.session_state.get("probe_data_path", "")
    if build_mode == "train_existing_data":
        existing_probe_data_path = st.text_input(
            "Existing probe data path",
            value=st.session_state.get("probe_data_path", ""),
        )
    if build_mode == "train_existing_data" and existing_data_candidates:
        picked_data = st.selectbox("Detected probe_data files", options=[""] + existing_data_candidates, key="pick_existing_data")
        if picked_data and st.button("Use Selected Data File", use_container_width=True):
            st.session_state["probe_data_path"] = picked_data
            st.rerun()

    load_artifact_path = st.session_state.get("artifact_path", "")
    if build_mode == "load_existing_artifact":
        load_artifact_path = st.text_input(
            "Existing artifact path",
            value=st.session_state.get("artifact_path", ""),
        )
    if build_mode == "load_existing_artifact" and existing_artifact_candidates:
        picked_art = st.selectbox("Detected probe artifacts", options=[""] + existing_artifact_candidates, key="pick_existing_artifact")
        if picked_art and st.button("Use Selected Artifact File", use_container_width=True):
            st.session_state["artifact_path"] = picked_art
            st.rerun()

    preflight_errors: list[str] = []
    preflight_warnings: list[str] = []

    if not str(gen_model_id).strip():
        preflight_errors.append("Model ID is required.")

    if build_mode == "generate_train":
        if not emotions:
            preflight_errors.append("Select at least one emotion.")
        if len(emotions) < 2:
            preflight_errors.append("Select at least 2 emotions for meaningful paper-faithful training.")
        if not selected_topics:
            preflight_errors.append("Select at least one topic.")
        est_items = _estimate_generation_items(
            n_topics=len(selected_topics),
            n_emotions=len(emotions),
            stories_per_topic_emotion=1,
            dialogues_per_topic_emotion=0,
            neutral_dialogues_per_topic=2,
        )
        st.caption(f"Estimated generated records: {est_items}")
        out_data_path = _resolve(out_data)
        out_artifact_path = _resolve(out_artifact)
        if out_artifact_path.exists():
            preflight_errors.append(f"Output exists: {out_artifact_path}")
        if out_data_path.exists() and not out_artifact_path.exists():
            preflight_warnings.append(
                f"Found existing probe data: {out_data_path}. Generate + Train will resume from this file."
            )
        if gen_backend == "transformers" and est_items > 500:
            preflight_warnings.append("Large run size. Consider a smaller topic subset for a quick validation pass.")
        if not use_full_topics:
            preflight_warnings.append("Topic override is enabled; this run is not fully paper-faithful.")
    elif build_mode == "train_existing_data":
        source_data_raw = str(existing_probe_data_path or st.session_state.get("probe_data_path", "")).strip()
        out_artifact_path = _resolve(out_artifact)
        if not source_data_raw:
            preflight_errors.append("Provide an existing probe data path.")
        else:
            source_data_path = _resolve(source_data_raw)
        if source_data_raw and not source_data_path.exists():
            preflight_errors.append(f"Probe data not found: {source_data_path}")
        if out_artifact_path.exists():
            preflight_errors.append(f"Output exists: {out_artifact_path}")
    else:
        source_artifact_raw = str(load_artifact_path or st.session_state.get("artifact_path", "")).strip()
        if not source_artifact_raw:
            preflight_errors.append("Provide an existing artifact path.")
        else:
            source_artifact = _resolve(source_artifact_raw)
            if not source_artifact.exists():
                preflight_errors.append(f"Artifact not found: {source_artifact}")

    if gen_backend == "transformers" and gen_device == "mps" and int(max_length) > 2048:
        preflight_warnings.append("High max length on MPS can cause memory pressure.")

    for msg in preflight_warnings:
        st.warning(msg)
    for msg in preflight_errors:
        st.error(msg)

    running = _job_is_running(job if isinstance(job, dict) else None)
    action_label = {
        "generate_train": "Start Generate + Train",
        "train_existing_data": "Start Train from Existing Data",
        "load_existing_artifact": "Load Existing Artifact",
    }[build_mode]

    c_action1, c_action2 = st.columns(2)
    with c_action1:
        if st.button(action_label, type="primary", use_container_width=True, disabled=running or bool(preflight_errors)):
            try:
                if build_mode == "generate_train":
                    new_job = _new_build_job(build_mode)
                    st.session_state["build_job"] = new_job
                    cfg = DataGenConfig(
                        emotions=list(emotions),
                        topics=list(selected_topics),
                        stories_per_topic_emotion=1,
                        dialogues_per_topic_emotion=0,
                        neutral_dialogues_per_topic=2,
                        seed=int(seed),
                        backend_name=gen_backend,
                        model_id=gen_model_id,
                        tokenizer_id=gen_model_id,
                        device=gen_device,
                        dtype=gen_dtype,
                        max_length=int(max_length),
                        max_new_tokens=int(max_new_tokens),
                        temperature=0.0,
                        generation_batch_size=int(generation_batch_size),
                        use_generation_cache=bool(generation_cache),
                        generation_cache_dir=str(generation_cache_dir),
                    )
                    _start_generate_train_job(
                        job=new_job,
                        cfg=cfg,
                        out_data=_resolve(out_data),
                        out_artifact=_resolve(out_artifact),
                        train_num_layers=int(train_num_layers),
                        train_hidden_size=int(train_hidden_size),
                    )
                    st.rerun()
                elif build_mode == "train_existing_data":
                    new_job = _new_build_job(build_mode)
                    st.session_state["build_job"] = new_job
                    _start_train_only_job(
                        job=new_job,
                        probe_data_path=_resolve(str(existing_probe_data_path or st.session_state.get("probe_data_path", "")).strip()),
                        out_artifact=_resolve(out_artifact),
                        backend_name=gen_backend,
                        model_id=gen_model_id,
                        tokenizer_id=gen_model_id,
                        device=gen_device,
                        dtype=gen_dtype,
                        max_length=int(max_length),
                        train_num_layers=int(train_num_layers),
                        train_hidden_size=int(train_hidden_size),
                    )
                    st.rerun()
                else:
                    artifact = read_json(_resolve(load_artifact_path or st.session_state.get("artifact_path", "")))
                    validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")
                    probe_names = [str(p["probe_name"]) for p in artifact.get("probes", [])]
                    st.session_state["active_artifact"] = artifact
                    st.session_state["artifact_path"] = str(_resolve(load_artifact_path or st.session_state.get("artifact_path", "")))
                    st.session_state["slider_weights"] = {name: 0.0 for name in probe_names}
                    st.session_state["chat_backend_cfg"] = {
                        "backend_name": gen_backend,
                        "model_id": str(artifact["model"]["model_id"]),
                        "tokenizer_id": str(artifact["model"]["tokenizer_id"]),
                        "device": gen_device,
                        "dtype": gen_dtype,
                        "max_length": int(max_length),
                        "max_new_tokens": 256,
                        "temperature": 0.0,
                        "num_layers": int(artifact["model"]["num_layers"]),
                        "hidden_size": int(artifact["model"]["hidden_size"]),
                    }
                    st.success(f"Loaded artifact with {len(probe_names)} probes.")
            except Exception as exc:
                st.error(_friendly_error(str(exc)))

    with c_action2:
        if st.button("Stop Running Job", use_container_width=True, disabled=not running):
            if isinstance(st.session_state.get("build_job"), dict):
                st.session_state["build_job"]["cancel_requested"] = True
                st.session_state["build_job"]["message"] = "Cancellation requested. Finishing current step..."
            st.rerun()

    job = st.session_state.get("build_job")
    if isinstance(job, dict):
        st.progress(int(round(float(job.get("progress", 0.0)) * 100)), text=str(job.get("message", "Running")))
        status = str(job.get("status", ""))
        if status == "running":
            st.caption("Build job running in background. Use refresh if progress looks stale.")
            if st.button("Refresh Status", use_container_width=False, key="refresh_build_status"):
                st.rerun()
        elif status == "completed":
            result = job.get("result") or {}
            st.success(
                f"Run complete. rows={int(result.get('rows_count', 0))}, probes={int(result.get('probe_count', 0))}"
            )
            st.markdown(
                f"<span class='pill'>data={result.get('data_path','')}</span>"
                f"<span class='pill'>artifact={result.get('artifact_path','')}</span>",
                unsafe_allow_html=True,
            )
        elif status == "cancelled":
            st.warning("Run cancelled.")
        elif status == "error":
            st.error(_friendly_error(str(job.get("error", ""))))

    if st.session_state["build_history"]:
        st.markdown("**Session Build History**")
        history = st.session_state["build_history"]
        selected_idx = st.selectbox(
            "Recent runs",
            options=list(range(len(history))),
            format_func=lambda i: (
                f"{history[i]['at']} | {history[i]['mode']} | {history[i]['status']} | "
                f"probes={history[i]['probe_count']} | rows={history[i]['rows_count']}"
            ),
        )
        entry = history[int(selected_idx)]
        st.caption(f"artifact: {entry.get('artifact_path','')}")
        if entry.get("artifact_path") and st.button("Load Artifact from Selected Run", use_container_width=True):
            try:
                path = _resolve(str(entry["artifact_path"]))
                artifact = read_json(path)
                validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")
                probe_names = [str(p["probe_name"]) for p in artifact.get("probes", [])]
                st.session_state["active_artifact"] = artifact
                st.session_state["artifact_path"] = str(path)
                st.session_state["slider_weights"] = {name: 0.0 for name in probe_names}
                st.success(f"Loaded artifact with {len(probe_names)} probes.")
            except Exception as exc:
                st.error(_friendly_error(str(exc)))

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("2) Slider Steering")

    artifact_path_input = st.text_input("Load artifact", value=st.session_state.get("artifact_path", default_artifact_path))
    if existing_artifact_candidates:
        quick_art = st.selectbox("Quick pick artifact", options=[""] + existing_artifact_candidates, key="quick_pick_artifact")
        if quick_art and st.button("Use Quick Pick", use_container_width=True):
            st.session_state["artifact_path"] = quick_art
            st.rerun()

    if st.button("Load Artifact", use_container_width=True):
        try:
            artifact = read_json(_resolve(artifact_path_input))
            validate_document(artifact, "probe_artifact.schema.json", context="probe_artifact")
            probe_names = [str(p["probe_name"]) for p in artifact.get("probes", [])]
            st.session_state["active_artifact"] = artifact
            st.session_state["artifact_path"] = artifact_path_input
            st.session_state["slider_weights"] = {name: 0.0 for name in probe_names}
            st.success(f"Loaded artifact with {len(probe_names)} probes")
        except Exception as exc:
            st.error(str(exc))

    artifact = st.session_state.get("active_artifact")
    if artifact:
        probe_names = [str(p["probe_name"]) for p in artifact.get("probes", [])]
        _sync_slider_state(st, probe_names)

        if st.button("Reset Sliders", use_container_width=True):
            for name in probe_names:
                st.session_state[f"slider__{name}"] = 0.0
            st.session_state["slider_weights"] = {name: 0.0 for name in probe_names}

        slider_cols = st.columns(2)
        for idx, name in enumerate(probe_names):
            with slider_cols[idx % 2]:
                current = float(st.session_state["slider_weights"].get(name, 0.0))
                st.slider(
                    f"{name}",
                    min_value=-0.1,
                    max_value=0.1,
                    value=current,
                    step=0.005,
                    key=f"slider__{name}",
                )

        weights = _current_weights(st, probe_names)
        _, selected_layer, magnitude = _combined_weights_vector(artifact, weights)
        st.markdown(
            f"<span class='pill'>selected_layer={selected_layer}</span>"
            f"<span class='pill'>active_sliders={sum(1 for v in weights.values() if abs(v) > 1e-9)}</span>"
            f"<span class='pill'>blend_magnitude={magnitude:.4f}</span>",
            unsafe_allow_html=True,
        )

        profile_path_value = st.text_input("Profile path (save/load)", value=default_profile_path)
        p1, p2 = st.columns(2)
        with p1:
            if st.button("Save Slider Profile", use_container_width=True):
                try:
                    out_path = _resolve(profile_path_value)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if out_path.exists():
                        raise ArtError(f"Output exists (immutable): {out_path}")
                    _save_slider_profile(out_path, artifact=artifact, weights=weights)
                    st.success(f"Saved profile -> {out_path}")
                except Exception as exc:
                    st.error(str(exc))
        with p2:
            if st.button("Load Profile", use_container_width=True):
                try:
                    in_path = _resolve(profile_path_value)
                    if not in_path.exists():
                        raise ArtError(f"Profile not found: {in_path}")
                    loaded_weights = _load_slider_profile_weights(in_path, artifact=artifact)
                    st.session_state["slider_weights"] = loaded_weights
                    for name, value in loaded_weights.items():
                        st.session_state[f"slider__{name}"] = float(value)
                    st.success(f"Loaded profile -> {in_path}")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
    else:
        st.info("Generate + Train first, or load an existing probe artifact.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("3) Live Chat (Baseline vs Steered)")

    if artifact:
        default_cfg = st.session_state.get("chat_backend_cfg", {})
        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            chat_backend = st.selectbox(
                "Chat backend",
                options=["transformers", "transformerlens", "mock"],
                index=(
                    1
                    if default_cfg.get("backend_name", "transformers") == "transformerlens"
                    else (2 if default_cfg.get("backend_name", "transformers") == "mock" else 0)
                ),
            )
        with cc2:
            chat_model_id = st.text_input("Chat model", value=str(default_cfg.get("model_id", artifact["model"]["model_id"])))
        with cc3:
            chat_device = st.selectbox("Chat device", options=["auto", "cpu", "mps", "cuda"], index=0)
        with cc4:
            chat_dtype = st.selectbox("Chat dtype", options=["auto", "float32", "float16", "bfloat16"], index=0)

        cc5, cc6, cc7 = st.columns(3)
        with cc5:
            chat_max_length = st.number_input("Chat max length", min_value=64, max_value=8192, value=int(default_cfg.get("max_length", DEFAULT_MAX_LENGTH)))
        with cc6:
            chat_max_new = st.number_input("Chat max new tokens", min_value=8, max_value=1024, value=int(default_cfg.get("max_new_tokens", 256)))
        with cc7:
            chat_temp = st.number_input("Chat temperature", min_value=0.0, max_value=2.0, value=float(default_cfg.get("temperature", 0.0)), step=0.05)

        user_msg = st.chat_input("Ask the model and observe slider steering effects")
        if user_msg:
            try:
                weights = _current_weights(st, [str(p["probe_name"]) for p in artifact.get("probes", [])])
                combined, layer, magnitude = _combined_weights_vector(artifact, weights)

                backend = _get_runtime_backend(
                    st,
                    backend_name=chat_backend,
                    model_id=chat_model_id,
                    tokenizer_id=chat_model_id,
                    device=chat_device,
                    dtype=chat_dtype,
                    num_layers=int(artifact["model"]["num_layers"]),
                    hidden_size=int(artifact["model"]["hidden_size"]),
                )

                baseline_resp, baseline_count = backend.generate(
                    user_msg,
                    max_new_tokens=int(chat_max_new),
                    temperature=float(chat_temp),
                    steering=None,
                    max_length=int(chat_max_length),
                )

                if magnitude > 1e-8:
                    steering = SteeringIntervention(vector=combined, layers=[layer], alpha=1.0)
                    steered_resp, steered_count = backend.generate(
                        user_msg,
                        max_new_tokens=int(chat_max_new),
                        temperature=float(chat_temp),
                        steering=steering,
                        max_length=int(chat_max_length),
                    )
                else:
                    steered_resp = baseline_resp
                    steered_count = baseline_count

                if magnitude > 1e-8:
                    baseline_score = _projection_score(
                        backend=backend,
                        prompt=user_msg,
                        response=baseline_resp,
                        response_token_count=baseline_count,
                        vector=combined,
                        layer=layer,
                        max_length=int(chat_max_length),
                    )
                    steered_score = _projection_score(
                        backend=backend,
                        prompt=user_msg,
                        response=steered_resp,
                        response_token_count=steered_count,
                        vector=combined,
                        layer=layer,
                        max_length=int(chat_max_length),
                    )
                else:
                    baseline_score = 0.0
                    steered_score = 0.0

                st.session_state["studio_chat_history"].append(
                    {
                        "user": user_msg,
                        "baseline": baseline_resp,
                        "steered": steered_resp,
                        "weights": {k: float(v) for k, v in weights.items()},
                        "blend_magnitude": magnitude,
                        "baseline_score": baseline_score,
                        "steered_score": steered_score,
                        "delta_score": float(steered_score - baseline_score),
                        "baseline_severity": _severity_bucket(baseline_score),
                        "steered_severity": _severity_bucket(steered_score),
                    }
                )
            except Exception as exc:
                st.error(_friendly_error(str(exc)))

        for item in reversed(st.session_state["studio_chat_history"][-8:]):
            st.markdown("---")
            st.markdown(f"**User**: {item['user']}")
            st.markdown(
                f"<span class='pill'>blend_magnitude={float(item['blend_magnitude']):.4f}</span>"
                f"<span class='pill'>baseline_score={float(item.get('baseline_score', 0.0)):+.4f}</span>"
                f"<span class='pill'>steered_score={float(item.get('steered_score', 0.0)):+.4f}</span>"
                f"<span class='pill'>delta={float(item.get('delta_score', 0.0)):+.4f}</span>"
                f"<span class='pill'>severity {int(item.get('baseline_severity', 1))}->{int(item.get('steered_severity', 1))}</span>",
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            with left:
                st.markdown("**Baseline**")
                st.write(item["baseline"])
            with right:
                st.markdown("**Steered**")
                st.write(item["steered"])
    else:
        st.info("No trained artifact loaded yet.")

    st.markdown("</div>", unsafe_allow_html=True)


def launch() -> None:
    app_path = Path(__file__).resolve()
    env = os.environ.copy()
    host = env.get("ART_UI_HOST", "0.0.0.0")
    port = str(env.get("ART_UI_PORT", "8501"))
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            host,
            "--server.port",
            port,
            "--server.headless",
            "true",
            "--server.fileWatcherType",
            "none",
            "--server.runOnSave",
            "false",
            "--server.enableCORS",
            env.get("ART_UI_ENABLE_CORS", "false"),
            "--server.enableXsrfProtection",
            env.get("ART_UI_ENABLE_XSRF", "false"),
            "--server.enableWebsocketCompression",
            env.get("ART_UI_ENABLE_WS_COMPRESSION", "false"),
            "--browser.gatherUsageStats",
            env.get("ART_UI_GATHER_USAGE_STATS", "false"),
            "--logger.level",
            env.get("ART_UI_LOG_LEVEL", "error"),
        ],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    render()
