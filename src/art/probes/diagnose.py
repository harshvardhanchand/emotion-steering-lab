"""Probe diagnosis pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from art.backends import create_backend
from art.constants import DEFAULT_MAX_LENGTH, SCHEMA_VERSION
from art.errors import ArtError
from art.repro import hash_object, utc_now_iso
from art.steering.engine import severity_from_score, zscore


def _prompt_from_case(case: dict[str, Any]) -> str:
    return str(case.get("prompt") or case.get("text") or "")


def _linear_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x_center = x - x.mean()
    y_center = values - values.mean()
    denom = float((x_center**2).sum())
    if denom <= 1e-12:
        return 0.0
    return float((x_center * y_center).sum() / denom)


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def run_diagnosis(
    *,
    probe_artifact: dict[str, Any],
    cases: list[dict[str, Any]],
    run_id: str,
    backend_name: str = "transformers",
    model_id: str = "",
    tokenizer_id: str = "",
    device: str = "auto",
    dtype: str = "auto",
    max_length: int = DEFAULT_MAX_LENGTH,
    max_new_tokens: int = 96,
    temperature: float = 0.0,
    include_layer_matrix: bool = False,
) -> list[dict[str, Any]]:
    probes = list(probe_artifact.get("probes", []))
    if not probes:
        raise ArtError("Probe artifact has no probes")

    selected_layer = int(probe_artifact["layer_selection"]["selected_layer"])
    all_probe_names = [str(p["probe_name"]) for p in probes]
    per_probe_vectors = {
        str(p["probe_name"]): np.asarray(p["vector"], dtype=np.float32)
        for p in probes
    }
    per_probe_mean = {
        str(p["probe_name"]): float(p.get("validation_metrics", {}).get("zscore_reference_mean", 0.0))
        for p in probes
    }
    per_probe_std = {
        str(p["probe_name"]): float(p.get("validation_metrics", {}).get("zscore_reference_std", 1.0))
        for p in probes
    }

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

    for name, vec in per_probe_vectors.items():
        if vec.shape[0] != backend.hidden_size:
            raise ArtError(
                f"Probe '{name}' vector dim {vec.shape[0]} does not match backend hidden size {backend.hidden_size}"
            )

    diag_cfg = {
        "run_id": run_id,
        "backend_name": backend_name,
        "model_id": model_id or str(model_block.get("model_id", "")),
        "tokenizer_id": tokenizer_id or str(model_block.get("tokenizer_id", "")),
        "device": device,
        "dtype": dtype,
        "max_length": max_length,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "include_layer_matrix": include_layer_matrix,
    }
    config_hash = hash_object(diag_cfg)
    model_hash = backend.model_hash()
    created_at = utc_now_iso()

    sample_rows: list[dict[str, Any]] = []
    per_sample_first20_slopes: dict[tuple[str, int], dict[str, float]] = {}

    for idx, case in enumerate(cases):
        prompt = _prompt_from_case(case)
        scenario_id = str(case.get("scenario_id") or case.get("record_id") or f"case_{idx:06d}")
        category = str(case.get("category") or case.get("emotion_label") or "generic")
        seed = int(case.get("seed", 42))

        response, _ = backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            steering=None,
            max_length=max_length,
        )

        human_segment = f"Human: {prompt}"
        assistant_prefix = f"{human_segment}\nAssistant:"
        full_text = f"{assistant_prefix} {response}".strip()

        hidden = backend.extract_hidden_states(full_text, max_length=max_length)
        if selected_layer < 0 or selected_layer >= hidden.shape[0]:
            raise ArtError(f"Selected layer {selected_layer} out of range for diagnosis")

        seq_len = int(hidden.shape[1])
        user_end = max(0, min(seq_len - 1, backend.token_count(human_segment, max_length=max_length) - 1))
        assistant_colon = max(0, min(seq_len - 1, backend.token_count(assistant_prefix, max_length=max_length) - 1))
        response_tokens = max(1, backend.token_count(response, max_length=max_length))
        resp_start = min(seq_len - 1, assistant_colon + 1)
        resp_n = min(20, response_tokens, max(1, seq_len - resp_start))
        resp_end = min(seq_len, resp_start + resp_n)

        positions: dict[str, dict[str, float]] = {
            "user_end": {},
            "assistant_colon": {},
            "assistant_response_mean_20": {},
        }
        scores_by_probe: dict[str, list[float]] = {}
        probe_slopes: dict[str, float] = {}
        position_by_layer: list[dict[str, Any]] = []

        selected_hidden = hidden[selected_layer, :, :]
        for probe_name in all_probe_names:
            vec = per_probe_vectors[probe_name]
            mean = per_probe_mean[probe_name]
            std = per_probe_std[probe_name]

            token_scores_raw = selected_hidden @ vec
            token_scores = np.asarray([zscore(float(v), mean=mean, std=std) for v in token_scores_raw], dtype=np.float64)

            positions["user_end"][probe_name] = float(token_scores[user_end])
            positions["assistant_colon"][probe_name] = float(token_scores[assistant_colon])
            positions["assistant_response_mean_20"][probe_name] = float(token_scores[resp_start:resp_end].mean())
            scores_by_probe[probe_name] = token_scores.astype(float).tolist()
            probe_slopes[probe_name] = _linear_slope(token_scores[resp_start:resp_end])

        if include_layer_matrix:
            layers_for_matrix = probe_artifact.get("extraction_config", {}).get("central_14_layers") or [selected_layer]
            for layer in layers_for_matrix:
                layer_idx = int(layer)
                if layer_idx < 0 or layer_idx >= hidden.shape[0]:
                    continue
                layer_hidden = hidden[layer_idx, :, :]
                pos_scores = {
                    "user_end": {},
                    "assistant_colon": {},
                    "assistant_response_mean_20": {},
                }
                for probe_name in all_probe_names:
                    vec = per_probe_vectors[probe_name]
                    mean = per_probe_mean[probe_name]
                    std = per_probe_std[probe_name]
                    token_scores_raw = layer_hidden @ vec
                    token_scores = np.asarray([zscore(float(v), mean=mean, std=std) for v in token_scores_raw], dtype=np.float64)
                    pos_scores["user_end"][probe_name] = float(token_scores[user_end])
                    pos_scores["assistant_colon"][probe_name] = float(token_scores[assistant_colon])
                    pos_scores["assistant_response_mean_20"][probe_name] = float(token_scores[resp_start:resp_end].mean())
                position_by_layer.append({"layer": layer_idx, "position_scores": pos_scores})

        response_scores = list(positions["assistant_response_mean_20"].values())
        inferred_severity = severity_from_score(max(response_scores) if response_scores else 0.0)
        final_severity = int(case.get("baseline_final_severity", case.get("final_severity", inferred_severity)))
        failure = final_severity >= 2

        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "sample",
            "run_id": run_id,
            "scenario_id": scenario_id,
            "category": category,
            "seed": seed,
            "failure": failure,
            "final_severity": final_severity,
            "selected_layer": selected_layer,
            "positions": positions,
            "trajectory_selected_layer": {
                "token_indices": list(range(seq_len)),
                "scores_by_probe": scores_by_probe,
            },
            "reproducibility": {
                "created_at": created_at,
                "config_hash": config_hash,
                "model_hash": model_hash,
            },
        }
        if include_layer_matrix:
            record["layer_position_matrix"] = position_by_layer
        sample_rows.append(record)
        per_sample_first20_slopes[(scenario_id, seed)] = probe_slopes

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        grouped[str(row["category"])].append(row)

    for category, rows in grouped.items():
        failed = [r for r in rows if bool(r["failure"])]
        passed = [r for r in rows if not bool(r["failure"])]
        feature_deltas: dict[str, dict[str, float]] = {}

        for probe_name in all_probe_names:
            failed_user = [float(r["positions"]["user_end"][probe_name]) for r in failed]
            passed_user = [float(r["positions"]["user_end"][probe_name]) for r in passed]

            failed_colon = [float(r["positions"]["assistant_colon"][probe_name]) for r in failed]
            passed_colon = [float(r["positions"]["assistant_colon"][probe_name]) for r in passed]

            failed_resp = [float(r["positions"]["assistant_response_mean_20"][probe_name]) for r in failed]
            passed_resp = [float(r["positions"]["assistant_response_mean_20"][probe_name]) for r in passed]

            failed_slopes = [
                per_sample_first20_slopes[(str(r["scenario_id"]), int(r["seed"]))][probe_name]
                for r in failed
            ]
            passed_slopes = [
                per_sample_first20_slopes[(str(r["scenario_id"]), int(r["seed"]))][probe_name]
                for r in passed
            ]

            feature_deltas[probe_name] = {
                "user_end_delta": _mean_or_zero(failed_user) - _mean_or_zero(passed_user),
                "assistant_colon_delta": _mean_or_zero(failed_colon) - _mean_or_zero(passed_colon),
                "assistant_response_mean_20_delta": _mean_or_zero(failed_resp) - _mean_or_zero(passed_resp),
                "first20_slope_delta": _mean_or_zero(failed_slopes) - _mean_or_zero(passed_slopes),
            }

        summary_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "comparison_summary",
                "run_id": run_id,
                "category": category,
                "failed_count": len(failed),
                "passed_count": len(passed),
                "feature_deltas": feature_deltas,
                "reproducibility": {
                    "created_at": created_at,
                    "config_hash": config_hash,
                    "model_hash": model_hash,
                },
            }
        )

    return sample_rows + summary_rows

