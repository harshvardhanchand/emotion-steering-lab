"""Probe training pipeline using model hidden activations."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from art.backends import create_backend
from art.constants import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_NUM_LAYERS,
    DEFAULT_POOLING_MIN_TOKEN_INDEX,
    PAPER_MODE_NAME,
    SCHEMA_VERSION,
    project_root,
)
from art.errors import ArtError
from art.repro import hash_object, utc_now_iso

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


def _selected_layer(num_layers: int) -> int:
    return max(0, int((2 * num_layers) / 3) - 1)


def _pool_hidden_states(hidden: np.ndarray, *, min_token_index: int) -> np.ndarray:
    """Pool token activations to [layers, hidden_size]."""
    seq_len = int(hidden.shape[1])
    if seq_len <= 0:
        raise ArtError("Cannot pool empty hidden-state sequence")
    start = min(max(0, min_token_index), seq_len - 1)
    return hidden[:, start:, :].mean(axis=1)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v
    return v / norm


def _compute_pc_basis(matrix: np.ndarray, *, variance_target: float) -> tuple[np.ndarray, int]:
    """Return top principal components explaining variance target."""
    if matrix.shape[0] < 2:
        return np.zeros((0, matrix.shape[1]), dtype=np.float32), 0

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    var = s**2
    total = float(var.sum())
    if total <= 1e-12:
        return np.zeros((0, matrix.shape[1]), dtype=np.float32), 0

    ratios = var / total
    csum = np.cumsum(ratios)
    k = int(np.searchsorted(csum, variance_target, side="left") + 1)
    k = max(1, min(k, vh.shape[0]))
    return vh[:k, :].astype(np.float32), k


def _project_out(v: np.ndarray, basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return v
    return v - (basis.T @ (basis @ v))


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return 0.5

    scores = np.concatenate([pos, neg]).astype(np.float64)
    labels = np.concatenate([np.ones_like(pos, dtype=np.int64), np.zeros_like(neg, dtype=np.int64)])

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.float64)

    i = 0
    while i < order.size:
        j = i
        while j + 1 < order.size and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    n_pos = float(pos.size)
    n_neg = float(neg.size)
    rank_sum_pos = float(ranks[labels == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def _activation_cache_root(path: str | Path | None) -> Path:
    if path is None:
        return (project_root() / "cache" / "activations").resolve()
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (project_root() / p).resolve()


def _activation_cache_key(
    *,
    model_hash: str,
    text: str,
    max_length: int,
    min_token_index: int,
) -> str:
    return hash_object(
        {
            "schema": "activation_cache_v1",
            "model_hash": model_hash,
            "text_hash": hash_object({"text": text}),
            "max_length": int(max_length),
            "pooling": "mean_over_tokens_ge_min_index",
            "min_token_index": int(min_token_index),
        }
    )


def _activation_cache_path(root: Path, key: str) -> Path:
    return root / f"{key}.npz"


def _load_cached_pooled(path: Path, *, num_layers: int, hidden_size: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            pooled = np.asarray(data["pooled"], dtype=np.float32)
    except Exception:
        return None
    if pooled.ndim != 2:
        return None
    if int(pooled.shape[0]) != int(num_layers) or int(pooled.shape[1]) != int(hidden_size):
        return None
    return pooled.astype(np.float32, copy=False)


def _write_cached_pooled(path: Path, pooled: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, pooled=pooled.astype(np.float32))
    tmp.replace(path)


def train_probe_artifact(
    *,
    records: list[dict[str, Any]],
    model_id: str,
    tokenizer_id: str,
    num_layers: int = DEFAULT_NUM_LAYERS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    backend_name: str = "transformers",
    device: str = "auto",
    dtype: str = "auto",
    max_length: int = DEFAULT_MAX_LENGTH,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCheck | None = None,
    backend: Any | None = None,
    extraction_batch_size: int = 8,
    use_activation_cache: bool = True,
    activation_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not records:
        raise ArtError("Probe training needs at least one record")

    def _emit(frac: float, message: str) -> None:
        if progress_callback is None:
            return
        progress_callback(min(1.0, max(0.0, float(frac))), message)

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise ArtError("Cancelled by user")

    _emit(0.0, "Loading training backend")
    _check_cancel()

    if backend is None:
        backend = create_backend(
            backend_name=backend_name,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            device=device,
            dtype=dtype,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
    _emit(0.05, "Backend ready")

    story_rows = [
        r
        for r in records
        if str(r.get("source_type")) == "story" and str(r.get("emotion_label")) != "neutral"
    ]
    if not story_rows:
        raise ArtError("Probe training needs non-neutral story records")
    distinct_emotions = sorted({str(r.get("emotion_label")) for r in story_rows})
    if len(distinct_emotions) < 2:
        raise ArtError("Probe training needs at least two non-neutral emotion labels")
    neutral_rows = [r for r in records if str(r.get("source_type")) == "neutral_dialogue"]
    if not neutral_rows:
        raise ArtError(
            "Probe training needs neutral_dialogue records for paper-faithful neutral PC removal"
        )

    # Probe fitting only requires non-neutral stories + neutral dialogues.
    # Emotional dialogues are retained in datasets for diagnostics/validation, but skipped here.
    extraction_rows = [
        r for r in records if str(r.get("source_type")) in {"story", "neutral_dialogue"}
    ]

    pooled_by_record_id: dict[str, np.ndarray] = {}
    total_records = max(1, len(extraction_rows))
    batch_size = max(1, int(extraction_batch_size))
    batch_extractor = getattr(backend, "extract_hidden_states_batch", None)
    model_hash = backend.model_hash()
    cache_root = _activation_cache_root(activation_cache_dir) if use_activation_cache else None
    cache_hits = 0
    cache_misses = 0
    done = 0

    pending_rows: list[tuple[dict[str, Any], int, Path | None]] = []
    for row in extraction_rows:
        _check_cancel()
        min_token_index = int(
            row.get("extraction_hints", {}).get(
                "min_token_index_for_pooling", DEFAULT_POOLING_MIN_TOKEN_INDEX
            )
        )
        cache_path: Path | None = None
        if cache_root is not None:
            key = _activation_cache_key(
                model_hash=model_hash,
                text=str(row["text"]),
                max_length=max_length,
                min_token_index=min_token_index,
            )
            cache_path = _activation_cache_path(cache_root, key)
            cached = _load_cached_pooled(
                cache_path,
                num_layers=int(backend.num_layers),
                hidden_size=int(backend.hidden_size),
            )
            if cached is not None:
                pooled_by_record_id[str(row["record_id"])] = cached
                cache_hits += 1
                done += 1
                _emit(0.05 + (0.55 * (done / total_records)), f"Extracted activations ({done}/{total_records})")
                continue
        cache_misses += 1
        pending_rows.append((row, min_token_index, cache_path))

    if callable(batch_extractor):
        for start in range(0, len(pending_rows), batch_size):
            _check_cancel()
            batch_items = pending_rows[start : start + batch_size]
            texts = [str(item[0]["text"]) for item in batch_items]
            hidden_batch = list(batch_extractor(texts, max_length=max_length))
            if len(hidden_batch) != len(batch_items):
                raise ArtError(
                    f"Batch extraction returned {len(hidden_batch)} items for {len(batch_items)} inputs"
                )
            for (row, min_token_index, cache_path), hidden in zip(batch_items, hidden_batch):
                if hidden.shape[0] != backend.num_layers:
                    raise ArtError(
                        f"Backend returned {hidden.shape[0]} layers but declared {backend.num_layers}"
                    )
                pooled = _pool_hidden_states(hidden, min_token_index=min_token_index)
                pooled_by_record_id[str(row["record_id"])] = pooled
                if cache_path is not None:
                    _write_cached_pooled(cache_path, pooled)
                done += 1
                _emit(0.05 + (0.55 * (done / total_records)), f"Extracted activations ({done}/{total_records})")
    else:
        for row, min_token_index, cache_path in pending_rows:
            _check_cancel()
            text = str(row["text"])
            hidden = backend.extract_hidden_states(text, max_length=max_length)
            if hidden.shape[0] != backend.num_layers:
                raise ArtError(
                    f"Backend returned {hidden.shape[0]} layers but declared {backend.num_layers}"
                )
            pooled = _pool_hidden_states(hidden, min_token_index=min_token_index)
            pooled_by_record_id[str(row["record_id"])] = pooled
            if cache_path is not None:
                _write_cached_pooled(cache_path, pooled)
            done += 1
            _emit(0.05 + (0.55 * (done / total_records)), f"Extracted activations ({done}/{total_records})")

    by_emotion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in story_rows:
        by_emotion[str(row["emotion_label"])].append(row)

    emotions = sorted(by_emotion)
    emotion_vectors_by_layer: dict[str, np.ndarray] = {}
    metrics: dict[str, float] = {}
    calibration_means: dict[str, float] = {}
    calibration_stds: dict[str, float] = {}

    train_rows = [r for r in story_rows if str(r.get("split")) == "train"] or story_rows
    emotion_means: dict[str, np.ndarray] = {}
    for emotion in emotions:
        pos_rows = [r for r in train_rows if str(r["emotion_label"]) == emotion]
        if not pos_rows:
            continue
        vectors = np.stack([pooled_by_record_id[str(r["record_id"])] for r in pos_rows], axis=0)
        emotion_means[emotion] = vectors.mean(axis=0)

    if not emotion_means:
        raise ArtError("No emotion vectors could be trained")

    mean_across_emotions = np.stack(list(emotion_means.values()), axis=0).mean(axis=0)
    neutral_by_layer: list[np.ndarray] = []
    neutral_stack = np.stack([pooled_by_record_id[str(r["record_id"])] for r in neutral_rows], axis=0)
    for layer_i in range(backend.num_layers):
        neutral_by_layer.append(neutral_stack[:, layer_i, :])

    pc_basis_by_layer: list[np.ndarray] = []
    removed_count_by_layer: list[int] = []
    total_layers = max(1, backend.num_layers)
    for layer_i in range(backend.num_layers):
        _check_cancel()
        basis, k = _compute_pc_basis(
            neutral_by_layer[layer_i],
            variance_target=0.5,
        )
        pc_basis_by_layer.append(basis)
        removed_count_by_layer.append(k)
        _emit(0.62 + (0.13 * ((layer_i + 1) / total_layers)), f"Computed neutral PCs ({layer_i + 1}/{total_layers} layers)")

    total_emotions = max(1, len(emotions))
    for emotion_idx, emotion in enumerate(emotions):
        _check_cancel()
        if emotion not in emotion_means:
            continue
        raw = emotion_means[emotion] - mean_across_emotions  # [layers, hidden]
        clean_layers: list[np.ndarray] = []
        for layer_i in range(backend.num_layers):
            projected = _project_out(raw[layer_i], pc_basis_by_layer[layer_i])
            clean_layers.append(_normalize(projected))
        emotion_vectors_by_layer[emotion] = np.stack(clean_layers, axis=0).astype(np.float32)
        _emit(0.75 + (0.12 * ((emotion_idx + 1) / total_emotions)), f"Built probe vectors ({emotion_idx + 1}/{total_emotions})")

    selected_layer = _selected_layer(backend.num_layers)
    created_at = datetime.now(timezone.utc).isoformat()
    central_14 = list(range(max(0, selected_layer - 7), min(backend.num_layers, selected_layer + 7)))

    for emotion_idx, emotion in enumerate(emotions):
        _check_cancel()
        if emotion not in emotion_vectors_by_layer:
            continue
        vec = emotion_vectors_by_layer[emotion][selected_layer]
        val_rows = [r for r in story_rows if str(r.get("split")) == "val"] or train_rows
        pos = np.asarray(
            [
                float(np.dot(pooled_by_record_id[str(r["record_id"])][selected_layer], vec))
                for r in val_rows
                if str(r["emotion_label"]) == emotion
            ],
            dtype=np.float64,
        )
        neg = np.asarray(
            [
                float(np.dot(pooled_by_record_id[str(r["record_id"])][selected_layer], vec))
                for r in val_rows
                if str(r["emotion_label"]) != emotion
            ],
            dtype=np.float64,
        )
        metrics[emotion] = _auroc(pos, neg)

        neutral_scores = np.asarray(
            [
                float(np.dot(pooled_by_record_id[str(r["record_id"])][selected_layer], vec))
                for r in neutral_rows
            ],
            dtype=np.float64,
        )
        if neutral_scores.size == 0:
            calibration_means[emotion] = 0.0
            calibration_stds[emotion] = 1.0
        else:
            calibration_means[emotion] = float(neutral_scores.mean())
            std = float(neutral_scores.std())
            calibration_stds[emotion] = std if std > 1e-6 else 1.0
        _emit(0.87 + (0.11 * ((emotion_idx + 1) / total_emotions)), f"Calibrated probes ({emotion_idx + 1}/{total_emotions})")

    topic_set = sorted({str(r.get("topic", "")) for r in records if str(r.get("topic", ""))})
    train_cfg = {
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "backend_name": backend_name,
        "device": device,
        "dtype": dtype,
        "max_length": max_length,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "extraction_batch_size": batch_size,
        "use_activation_cache": bool(use_activation_cache),
        "activation_cache_dir": str(cache_root) if cache_root is not None else "",
        "activation_cache_hits": int(cache_hits),
        "activation_cache_misses": int(cache_misses),
        "pooling_min_token_index": DEFAULT_POOLING_MIN_TOKEN_INDEX,
        "variance_explained_target": 0.5,
    }
    data_hash = hash_object(records)
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": f"probe_{created_at.replace(':', '').replace('-', '')}",
        "created_at": created_at,
        "model": {
            "model_id": backend.model_id,
            "tokenizer_id": backend.tokenizer_id,
            "num_layers": backend.num_layers,
            "hidden_size": backend.hidden_size,
        },
        "method": {
            "name": "story_mean_difference",
            "paper": PAPER_MODE_NAME,
            "citations": [
                "https://www.anthropic.com/research/emotion-concepts-function",
                "https://transformer-circuits.pub/2026/emotions/index.html",
            ],
        },
        "dataset_spec": {
            "num_topics": len(topic_set) if topic_set else 1,
            "stories_per_topic_per_emotion": 1,
            "emotion_labels": emotions,
            "pooling_min_token_index": DEFAULT_POOLING_MIN_TOKEN_INDEX,
            "neutral_dataset_id": "neutral_dialogues_v1",
        },
        "extraction_config": {
            "layers_mode": "all_layers",
            "central_14_layers": central_14,
            "token_positions_for_diagnosis": [
                "user_end",
                "assistant_colon",
                "assistant_response_mean_20",
            ],
            "pooling": "mean_over_tokens_ge_min_index",
        },
        "preprocessing_config": {
            "subtract_mean_across_emotions": True,
            "neutral_pc_removal": {
                "enabled": True,
                "variance_explained_target": 0.5,
                "removed_components_count": removed_count_by_layer[selected_layer],
            },
            "normalization": "zscore_reference",
        },
        "training_config": {
            "estimator": "mean_difference",
            "backend": backend.backend_name,
            "max_length": max_length,
        },
        "layer_selection": {
            "strategy": "two_thirds_depth",
            "selected_layer": selected_layer,
            "selection_metric": "auroc",
        },
        "calibration": {
            "method": "zscore_reference",
            "reference_dataset_id": "neutral_dialogues_v1",
            "parameters": {
                "neutral_pc_removed_by_layer": removed_count_by_layer,
            },
        },
        "reproducibility": {
            "created_at": utc_now_iso(),
            "config_hash": hash_object(train_cfg),
            "model_hash": backend.model_hash(),
            "data_hash": data_hash,
        },
        "probes": [
            {
                "probe_name": emotion,
                "selected_layer": selected_layer,
                "vector": emotion_vectors_by_layer[emotion][selected_layer].astype(float).tolist(),
                "per_layer_vectors": [
                    {
                        "layer": layer_i,
                        "vector": emotion_vectors_by_layer[emotion][layer_i].astype(float).tolist(),
                    }
                    for layer_i in range(backend.num_layers)
                ],
                "validation_metrics": {
                    "auroc": metrics.get(emotion, 0.5),
                    "zscore_reference_mean": calibration_means.get(emotion, 0.0),
                    "zscore_reference_std": calibration_stds.get(emotion, 1.0),
                },
            }
            for emotion in emotion_vectors_by_layer
        ],
    }

    # keep stable order for tests
    artifact["probes"] = sorted(artifact["probes"], key=lambda p: str(p["probe_name"]))
    _emit(1.0, "Training complete")
    return artifact
