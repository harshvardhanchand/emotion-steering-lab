"""TransformerLens backend for activation extraction and optional steering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from art.backends.base import SteeringIntervention
from art.errors import ArtError


def _pick_device(torch_mod: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_mod.cuda.is_available():
        return "cuda"
    if hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(torch_mod: Any, dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    if dtype == "float32":
        return torch_mod.float32
    if dtype == "float16":
        return torch_mod.float16
    if dtype == "bfloat16":
        return torch_mod.bfloat16
    raise ArtError(f"Unsupported dtype: {dtype}")


@dataclass
class TransformerLensBackend:
    """Backend powered by transformer_lens HookedTransformer."""

    model_id: str
    tokenizer_id: str
    device: str = "auto"
    dtype: str = "auto"
    backend_name: str = "transformerlens"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformer_lens import HookedTransformer, utils as tl_utils
        except ImportError as exc:
            raise ArtError(
                "transformerlens backend requires `transformer-lens`. "
                "Install with: uv sync --extra hf --extra tl"
            ) from exc

        self._torch = torch
        self._tl_utils = tl_utils
        self.device = _pick_device(torch, self.device)
        self._torch_dtype = _resolve_dtype(torch, self.dtype)

        load_err: Exception | None = None
        model: Any | None = None
        for kwargs in (
            {"device": self.device},
            {},
        ):
            try:
                model = HookedTransformer.from_pretrained(self.model_id, **kwargs)
                break
            except Exception as exc:  # pragma: no cover - environment/model dependent
                load_err = exc
                model = None
        if model is None:
            raise ArtError(
                f"Failed to load TransformerLens model '{self.model_id}'. "
                "Try a TransformerLens-supported model name."
            ) from load_err

        self.model = model
        if self._torch_dtype != "auto":
            self.model = self.model.to(dtype=self._torch_dtype)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.num_layers = int(getattr(self.model.cfg, "n_layers"))
        self.hidden_size = int(getattr(self.model.cfg, "d_model"))

        cfg_text = repr(self.model.cfg)
        raw = f"{self.backend_name}|{self.model_id}|{self.tokenizer_id}|{cfg_text}"
        self._model_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _to_tokens(self, text: str, *, max_length: int) -> Any:
        tokens = self.model.to_tokens(text, prepend_bos=True)
        if int(tokens.shape[1]) > max_length:
            tokens = tokens[:, -max_length:]
        return tokens.to(self.device)

    def _resid_name(self, cache: Any, layer: int) -> str:
        for kind in ("resid_post", "resid_pre"):
            name = self._tl_utils.get_act_name(kind, layer)
            if name in cache:
                return name
        raise ArtError(f"TransformerLens cache missing residual activations for layer {layer}")

    def extract_hidden_states(self, text: str, *, max_length: int) -> np.ndarray:
        torch = self._torch
        tokens = self._to_tokens(text, max_length=max_length)
        with torch.inference_mode():
            _, cache = self.model.run_with_cache(tokens, return_type="logits")

        seq_len = int(tokens.shape[1])
        layers: list[np.ndarray] = []
        for layer in range(self.num_layers):
            name = self._resid_name(cache, layer)
            resid = cache[name]
            layers.append(resid[0, :seq_len, :].detach().float().cpu().numpy())
        return np.stack(layers, axis=0)

    def extract_hidden_states_batch(self, texts: list[str], *, max_length: int) -> list[np.ndarray]:
        if not texts:
            return []
        # Keep backend behavior stable and memory-safe across TL versions.
        return [self.extract_hidden_states(t, max_length=max_length) for t in texts]

    def token_count(self, text: str, *, max_length: int) -> int:
        tokens = self._to_tokens(text, max_length=max_length)
        return max(1, int(tokens.shape[1]))

    def model_hash(self) -> str:
        return self._model_hash

    def _fwd_hooks(self, steering: SteeringIntervention) -> list[tuple[str, Any]]:
        if steering.vector.shape[0] != self.hidden_size:
            raise ArtError(
                f"Steering vector size {steering.vector.shape[0]} does not match hidden size {self.hidden_size}"
            )
        torch = self._torch
        vec = torch.tensor(steering.vector, device=self.device, dtype=torch.float32)
        vec_norm = torch.linalg.norm(vec).clamp_min(1e-6)
        unit = vec / vec_norm

        hooks: list[tuple[str, Any]] = []
        for layer_idx in steering.layers:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ArtError(f"Invalid steering layer {layer_idx} for model with {self.num_layers} layers")
            name = self._tl_utils.get_act_name("resid_post", layer_idx)

            def _hook(resid: Any, _hook: Any, unit_vec: Any = unit, alpha: float = float(steering.alpha)) -> Any:
                steer_vec = unit_vec.to(device=resid.device, dtype=resid.dtype).view(1, 1, -1)
                tail = resid[:, -1:, :]
                norms = self._torch.linalg.norm(tail, dim=-1, keepdim=True).clamp_min(1e-6)
                resid = resid.clone()
                resid[:, -1:, :] = tail + (alpha * norms * steer_vec)
                return resid

            hooks.append((name, _hook))
        return hooks

    def _decode_new_tokens(self, generated: Any, prompt_len: int) -> tuple[str, int]:
        new_tokens = generated[:, prompt_len:]
        token_count = int(new_tokens.shape[1])
        if token_count <= 0:
            return "", 1
        ids = new_tokens[0].detach().cpu().tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        return text, max(1, token_count)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        steering: SteeringIntervention | None = None,
        max_length: int = 2048,
    ) -> tuple[str, int]:
        torch = self._torch
        tokens = self._to_tokens(prompt, max_length=max_length)
        prompt_len = int(tokens.shape[1])
        max_steps = min(int(max_new_tokens), max(0, int(max_length) - prompt_len))
        if max_steps <= 0:
            return "", 1
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        hooks = self._fwd_hooks(steering) if steering is not None and float(steering.alpha) != 0.0 else []

        cur = tokens
        for _ in range(max_steps):
            with torch.inference_mode():
                if hooks:
                    logits = self.model.run_with_hooks(cur, return_type="logits", fwd_hooks=hooks)
                else:
                    logits = self.model(cur, return_type="logits")

            next_logits = logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_logits / float(temperature), dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            cur = torch.cat([cur, next_token], dim=1)
            if eos_id is not None and bool((next_token == int(eos_id)).all()):
                break

        return self._decode_new_tokens(cur, prompt_len)

    def generate_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        steering: SteeringIntervention | None = None,
        max_length: int = 2048,
    ) -> list[tuple[str, int]]:
        if not prompts:
            return []
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
