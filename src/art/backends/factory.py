"""Backend factory."""

from __future__ import annotations

from art.backends.base import ModelBackend
from art.backends.mock_backend import MockBackend
from art.backends.transformerlens_backend import TransformerLensBackend
from art.backends.transformers_backend import TransformersBackend
from art.errors import ArtError


def create_backend(
    *,
    backend_name: str,
    model_id: str,
    tokenizer_id: str,
    device: str = "auto",
    dtype: str = "auto",
    num_layers: int | None = None,
    hidden_size: int | None = None,
) -> ModelBackend:
    if backend_name == "transformers":
        return TransformersBackend(
            model_id=model_id,
            tokenizer_id=tokenizer_id or model_id,
            device=device,
            dtype=dtype,
        )

    if backend_name == "transformerlens":
        return TransformerLensBackend(
            model_id=model_id,
            tokenizer_id=tokenizer_id or model_id,
            device=device,
            dtype=dtype,
        )

    if backend_name == "mock":
        return MockBackend(
            model_id=model_id,
            tokenizer_id=tokenizer_id or model_id,
            num_layers=num_layers or 12,
            hidden_size=hidden_size or 64,
        )

    raise ArtError(f"Unsupported backend: {backend_name}")
