"""Deterministic mock backend for tests and contract checks."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from art.backends.base import SteeringIntervention

TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")
N_STORIES_RE = re.compile(r"Write\\s+(\\d+)\\s+different", re.IGNORECASE)


def _seed_for(*parts: str) -> int:
    joined = "||".join(parts)
    return int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16], 16)


@dataclass
class MockBackend:
    """Simple deterministic backend to keep CI lightweight."""

    model_id: str
    tokenizer_id: str
    num_layers: int = 12
    hidden_size: int = 64
    backend_name: str = "mock"

    def token_count(self, text: str, *, max_length: int) -> int:
        return max(1, min(max_length, len(TOKEN_RE.findall(text))))

    def model_hash(self) -> str:
        raw = f"{self.backend_name}|{self.model_id}|{self.tokenizer_id}|{self.num_layers}|{self.hidden_size}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _generate_story_blocks(self, prompt: str, n: int) -> str:
        topic = "unspecified topic"
        for line in prompt.splitlines():
            if line.strip().startswith("Topic:"):
                topic = line.split("Topic:", 1)[1].strip() or topic
                break
        chunks: list[str] = []
        for i in range(1, n + 1):
            chunks.append(
                f"[story {i}]\n"
                f"A short scene about {topic}. The character reacts through behavior, pace, and body language."
            )
        return "\n\n".join(chunks)

    def _generate_dialogue_blocks(self, prompt: str, n: int) -> str:
        topic = "unspecified topic"
        for line in prompt.splitlines():
            if line.strip().startswith("Topic:"):
                topic = line.split("Topic:", 1)[1].strip() or topic
                break
        chunks: list[str] = []
        for i in range(1, n + 1):
            chunks.append(
                f"[dialogue {i}]\n\n"
                f"Person: Help me reason about this topic: {topic}.\n\n"
                "AI: I will provide a structured plan and key tradeoffs.\n\n"
                "Person: Add concrete next steps.\n\n"
                "AI: Step 1 define constraints. Step 2 evaluate options. Step 3 choose and review."
            )
        return "\n\n".join(chunks)

    def extract_hidden_states(self, text: str, *, max_length: int) -> np.ndarray:
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            tokens = ["<empty>"]
        tokens = tokens[: max(1, min(max_length, len(tokens)))]

        out = np.zeros((self.num_layers, len(tokens), self.hidden_size), dtype=np.float32)
        for layer in range(self.num_layers):
            for i, token in enumerate(tokens):
                seed = _seed_for(self.model_id, str(layer), str(i), token)
                rng = np.random.default_rng(seed)
                out[layer, i, :] = rng.standard_normal(self.hidden_size).astype(np.float32)
        return out

    def extract_hidden_states_batch(self, texts: list[str], *, max_length: int) -> list[np.ndarray]:
        return [self.extract_hidden_states(text, max_length=max_length) for text in texts]

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        steering: SteeringIntervention | None = None,
        max_length: int = 2048,
    ) -> tuple[str, int]:
        del temperature, max_length
        n = 1
        m = N_STORIES_RE.search(prompt)
        if m:
            n = max(1, int(m.group(1)))

        if "The story should follow a character who is feeling" in prompt:
            response = self._generate_story_blocks(prompt, n)
            token_count = max(1, len(TOKEN_RE.findall(response)))
            return response, token_count

        if "The dialogue should be between two characters" in prompt:
            response = self._generate_dialogue_blocks(prompt, n)
            token_count = max(1, len(TOKEN_RE.findall(response)))
            return response, token_count

        words = TOKEN_RE.findall(prompt)
        if not words:
            words = ["request"]
        words = words[: min(len(words), max_new_tokens)]
        response = " ".join(words)

        if steering and steering.alpha != 0:
            tag = "amplified" if steering.alpha > 0 else "damped"
            response = f"{tag}: {response}"

        token_count = self.token_count(response, max_length=max_new_tokens * 4)
        return response, token_count

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        steering: SteeringIntervention | None = None,
        max_length: int = 2048,
    ) -> list[tuple[str, int]]:
        return [
            self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                steering=steering,
                max_length=max_length,
            )
            for prompt in prompts
        ]
