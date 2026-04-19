"""Steering run orchestration with causal intervention."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from art.backends import SteeringIntervention, create_backend
from art.constants import DEFAULT_MAX_LENGTH, SCHEMA_VERSION
from art.errors import ArtError
from art.repro import hash_object, utc_now_iso
from art.steering.engine import score_projection, severity_from_score, zscore


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v
    return v / norm


def _prompt_from_case(case: dict[str, Any]) -> str:
    return str(case.get("prompt") or case.get("text") or "")


def _score_response(
    *,
    backend: Any,
    prompt: str,
    response: str,
    response_token_count: int,
    layer: int,
    vector: np.ndarray,
    max_length: int,
) -> float:
    convo = f"Human: {prompt}\nAssistant: {response}"
    hidden = backend.extract_hidden_states(convo, max_length=max_length)
    if layer < 0 or layer >= hidden.shape[0]:
        raise ArtError(f"Selected layer {layer} is out of range for hidden states")

    seq_len = int(hidden.shape[1])
    response_tokens = max(1, min(response_token_count, seq_len))
    start = max(0, seq_len - response_tokens)
    first_n = min(20, seq_len - start)
    end = start + max(1, first_n)
    pooled = hidden[layer, start:end, :].mean(axis=0)
    return score_projection(pooled, vector)


def run_steering(
    *,
    probe_artifact: dict[str, Any],
    probe_name: str,
    alpha: float,
    cases: list[dict[str, Any]],
    base_run_id: str,
    steer_run_id: str,
    scope: str,
    backend_name: str = "transformers",
    model_id: str = "",
    tokenizer_id: str = "",
    device: str = "auto",
    dtype: str = "auto",
    max_length: int = DEFAULT_MAX_LENGTH,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    steered_layers: list[int] | None = None,
) -> list[dict[str, Any]]:
    probes = {str(p["probe_name"]): p for p in probe_artifact.get("probes", [])}
    if probe_name not in probes:
        raise ArtError(f"Probe '{probe_name}' not found in artifact")

    probe = probes[probe_name]
    vector = np.asarray(probe["vector"], dtype=np.float32)
    vector = _normalize(vector).astype(np.float32)
    if steered_layers is not None and len(steered_layers) > 0:
        layers = [int(x) for x in steered_layers]
    else:
        layers = [int(probe.get("selected_layer", probe_artifact["layer_selection"]["selected_layer"]))]

    model_block = probe_artifact.get("model", {})
    backend = create_backend(
        backend_name=backend_name,
        model_id=model_id or str(model_block.get("model_id", "")),
        tokenizer_id=tokenizer_id or str(model_block.get("tokenizer_id", "")),
        device=device,
        dtype=dtype,
        num_layers=int(model_block.get("num_layers", 0) or 0) or None,
        hidden_size=int(model_block.get("hidden_size", 0) or 0) or None,
    )

    if backend.hidden_size != vector.shape[0]:
        raise ArtError(
            f"Probe vector dim ({vector.shape[0]}) does not match backend hidden size ({backend.hidden_size})"
        )

    expected_model_hash = str(probe_artifact.get("reproducibility", {}).get("model_hash", "")).strip()
    actual_model_hash = str(backend.model_hash()).strip()
    if expected_model_hash and actual_model_hash != expected_model_hash:
        raise ArtError(
            "Probe artifact model_hash mismatch. "
            f"artifact={expected_model_hash}, runtime={actual_model_hash}. "
            "Use the same model/tokenizer/backend configuration used for probe training."
        )

    z_mean = float(probe.get("validation_metrics", {}).get("zscore_reference_mean", 0.0))
    z_std = float(probe.get("validation_metrics", {}).get("zscore_reference_std", 1.0))
    run_cfg = {
        "probe_name": probe_name,
        "alpha": alpha,
        "steered_layers": layers,
        "scope": scope,
        "backend_name": backend_name,
        "model_id": model_id or str(model_block.get("model_id", "")),
        "tokenizer_id": tokenizer_id or str(model_block.get("tokenizer_id", "")),
        "device": device,
        "dtype": dtype,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }
    reproducibility = {
        "created_at": utc_now_iso(),
        "config_hash": hash_object(run_cfg),
        "model_hash": backend.model_hash(),
    }

    prepared: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        prompt = _prompt_from_case(case)
        scenario_id = str(case.get("scenario_id") or case.get("record_id") or f"case_{idx:06d}")
        category = str(case.get("category") or case.get("emotion_label") or "generic")
        seed = int(case.get("seed", 42))

        baseline_resp, baseline_count = backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            steering=None,
            max_length=max_length,
        )
        baseline_raw = _score_response(
            backend=backend,
            prompt=prompt,
            response=baseline_resp,
            response_token_count=baseline_count,
            layer=layers[0],
            vector=vector,
            max_length=max_length,
        )
        baseline_score = zscore(baseline_raw, mean=z_mean, std=z_std)
        baseline_final = int(case.get("baseline_final_severity", severity_from_score(baseline_score)))
        baseline_failure = baseline_final >= 2

        prepared.append(
            {
                "prompt": prompt,
                "scenario_id": scenario_id,
                "category": category,
                "seed": seed,
                "baseline_final": baseline_final,
                "baseline_failure": baseline_failure,
                "baseline_score": baseline_score,
            }
        )

    if scope == "failures_only":
        prepared = [row for row in prepared if bool(row["baseline_failure"])]
        if not prepared:
            raise ArtError("No baseline failures found for scope=failures_only")

    intervention = SteeringIntervention(vector=vector, layers=layers, alpha=alpha)
    sample_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in prepared:
        prompt = str(item["prompt"])
        scenario_id = str(item["scenario_id"])
        category = str(item["category"])
        seed = int(item["seed"])
        baseline_final = int(item["baseline_final"])
        baseline_failure = bool(item["baseline_failure"])
        baseline_score = float(item["baseline_score"])

        steered_resp, steered_count = backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            steering=intervention,
            max_length=max_length,
        )
        steered_raw = _score_response(
            backend=backend,
            prompt=prompt,
            response=steered_resp,
            response_token_count=steered_count,
            layer=layers[0],
            vector=vector,
            max_length=max_length,
        )
        steered_score = zscore(steered_raw, mean=z_mean, std=z_std)
        steered_final = int(severity_from_score(steered_score))
        steered_failure = steered_final >= 2

        row = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "sample",
            "base_run_id": base_run_id,
            "steer_run_id": steer_run_id,
            "scope": scope,
            "scenario_id": scenario_id,
            "category": category,
            "seed": seed,
            "probe_name": probe_name,
            "alpha": alpha,
            "steered_layers": layers,
            "baseline_final_severity": baseline_final,
            "steered_final_severity": steered_final,
            "baseline_failure": baseline_failure,
            "steered_failure": steered_failure,
            "delta_failure": int(steered_failure) - int(baseline_failure),
            "quality_delta": float(steered_score - baseline_score),
            "status": "ok",
            "reproducibility": reproducibility,
        }
        sample_rows.append(row)
        grouped[category].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for category, rows in grouped.items():
        n = len(rows)
        base_fail = sum(1 for r in rows if r["baseline_failure"]) / n
        steer_fail = sum(1 for r in rows if r["steered_failure"]) / n
        base_crit = sum(1 for r in rows if int(r["baseline_final_severity"]) == 3) / n
        steer_crit = sum(1 for r in rows if int(r["steered_final_severity"]) == 3) / n

        aggregate_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "aggregate",
                "base_run_id": base_run_id,
                "steer_run_id": steer_run_id,
                "scope": scope,
                "category": category,
                "probe_name": probe_name,
                "alpha": alpha,
                "sample_count": n,
                "baseline_failure_rate": base_fail,
                "steered_failure_rate": steer_fail,
                "delta_failure_rate": steer_fail - base_fail,
                "baseline_critical_rate": base_crit,
                "steered_critical_rate": steer_crit,
                "delta_critical_rate": steer_crit - base_crit,
                "reproducibility": reproducibility,
            }
        )

    return sample_rows + aggregate_rows
