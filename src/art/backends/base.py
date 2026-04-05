"""Backend interfaces for activation extraction and generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SteeringIntervention:
    """Inference-time residual intervention config."""

    vector: np.ndarray
    layers: list[int]
    alpha: float


class ModelBackend(Protocol):
    """Backend contract for probe training and steering runs."""

    backend_name: str
    model_id: str
    tokenizer_id: str
    num_layers: int
    hidden_size: int

    def extract_hidden_states(self, text: str, *, max_length: int) -> np.ndarray:
        """Return hidden states as [num_layers, seq_len, hidden_size]."""

    def token_count(self, text: str, *, max_length: int) -> int:
        """Return token count under backend tokenizer."""

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        steering: SteeringIntervention | None = None,
        max_length: int = 2048,
    ) -> tuple[str, int]:
        """Return response text and generated token count."""

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        steering: SteeringIntervention | None = None,
        max_length: int = 2048,
    ) -> list[tuple[str, int]]:
        """Return response text/token counts for a prompt batch."""

    def model_hash(self) -> str:
        """Return deterministic model fingerprint hash."""
