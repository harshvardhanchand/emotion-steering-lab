"""Transformers backend for activation extraction and causal steering."""

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


def _decoder_layers(model: Any) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return list(model.gpt_neox.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return list(model.transformer.blocks)
    raise ArtError("Unsupported model architecture: cannot find decoder layers")


@dataclass
class TransformersBackend:
    """Backend powered by Hugging Face Transformers."""

    model_id: str
    tokenizer_id: str
    device: str = "auto"
    dtype: str = "auto"
    backend_name: str = "transformers"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ArtError(
                "transformers backend requires `transformers` and `torch`. "
                "Install with: uv sync --extra hf"
            ) from exc

        self._torch = torch
        self.device = _pick_device(torch, self.device)
        torch_dtype = _resolve_dtype(torch, self.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id or self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

        self._layers = _decoder_layers(self.model)
        self.num_layers = len(self._layers)
        self.hidden_size = int(getattr(self.model.config, "hidden_size"))
        self.model_revision = str(getattr(self.model.config, "_commit_hash", "") or "")
        cfg_json = self.model.config.to_json_string(use_diff=False)
        raw = f"{self.backend_name}|{self.model_id}|{self.tokenizer_id}|{self.model_revision}|{cfg_json}"
        self._model_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _encode(self, text: str, *, max_length: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _encode_batch(self, texts: list[str], *, max_length: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def extract_hidden_states(self, text: str, *, max_length: int) -> np.ndarray:
        torch = self._torch
        batch = self._encode(text, max_length=max_length)
        with torch.inference_mode():
            out = self.model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)

        hidden_states = out.hidden_states
        if not hidden_states or len(hidden_states) < 2:
            raise ArtError("Model did not return hidden states")

        # Drop embedding layer and keep decoder layers only.
        stack = torch.stack(list(hidden_states[1:]), dim=0)  # [layers, batch, seq, hidden]
        return stack[:, 0, :, :].detach().float().cpu().numpy()

    def extract_hidden_states_batch(self, texts: list[str], *, max_length: int) -> list[np.ndarray]:
        if not texts:
            return []
        torch = self._torch
        batch = self._encode_batch(texts, max_length=max_length)
        with torch.inference_mode():
            out = self.model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)

        hidden_states = out.hidden_states
        if not hidden_states or len(hidden_states) < 2:
            raise ArtError("Model did not return hidden states")

        stack = torch.stack(list(hidden_states[1:]), dim=0)  # [layers, batch, seq, hidden]
        mask = batch.get("attention_mask")
        seq_max = int(stack.shape[2])
        outputs: list[np.ndarray] = []
        for i in range(len(texts)):
            if mask is None:
                seq_len = seq_max
            else:
                seq_len = int(mask[i].sum().item())
                seq_len = max(1, min(seq_max, seq_len))
            outputs.append(stack[:, i, :seq_len, :].detach().float().cpu().numpy())
        return outputs

    def token_count(self, text: str, *, max_length: int) -> int:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        ids = encoded.get("input_ids", [])
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            return max(1, len(ids[0]))
        if isinstance(ids, list):
            return max(1, len(ids))
        return 1

    def model_hash(self) -> str:
        return self._model_hash

    def _register_hooks(self, steering: SteeringIntervention) -> list[Any]:
        if steering.vector.shape[0] != self.hidden_size:
            raise ArtError(
                f"Steering vector size {steering.vector.shape[0]} does not match hidden size {self.hidden_size}"
            )

        torch = self._torch
        vec = torch.tensor(steering.vector, device=self.device, dtype=torch.float32)
        vec_norm = torch.linalg.norm(vec).clamp_min(1e-6)
        unit = vec / vec_norm
        handles: list[Any] = []

        for layer_idx in steering.layers:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ArtError(f"Invalid steering layer {layer_idx} for model with {self.num_layers} layers")

            layer = self._layers[layer_idx]

            def _hook(_module: Any, _inputs: Any, output: Any, unit_vec: Any = unit) -> Any:
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden is None:
                    return output

                steer_vec = unit_vec.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                tail = hidden[:, -1:, :]
                norms = torch.linalg.norm(tail, dim=-1, keepdim=True).clamp_min(1e-6)
                steered_tail = tail + (steering.alpha * norms * steer_vec)
                hidden = hidden.clone()
                hidden[:, -1:, :] = steered_tail

                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                return hidden

            handles.append(layer.register_forward_hook(_hook))

        return handles

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
        batch = self._encode(prompt, max_length=max_length)
        input_len = int(batch["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "eos_token_id": int(self.tokenizer.eos_token_id),
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False

        handles: list[Any] = []
        try:
            if steering and steering.alpha != 0:
                handles = self._register_hooks(steering)

            with torch.inference_mode():
                out = self.model.generate(**batch, **gen_kwargs)
        finally:
            for handle in handles:
                handle.remove()

        generated_ids = out[0, input_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        token_count = int(generated_ids.shape[0])
        return response, max(1, token_count)

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
        # Keep steering semantics identical to single-call path.
        if steering and steering.alpha != 0:
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

        torch = self._torch
        batch = self._encode_batch(prompts, max_length=max_length)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            input_lens = [int(batch["input_ids"].shape[1])] * len(prompts)
        else:
            input_lens = [int(x) for x in attention_mask.sum(dim=1).tolist()]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": int(self.tokenizer.pad_token_id),
            "eos_token_id": int(self.tokenizer.eos_token_id),
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            out = self.model.generate(**batch, **gen_kwargs)

        pad_id = int(self.tokenizer.pad_token_id)
        results: list[tuple[str, int]] = []
        for i in range(len(prompts)):
            start = max(0, min(int(out.shape[1]), int(input_lens[i])))
            generated_ids = out[i, start:]
            non_pad = generated_ids[generated_ids != pad_id]
            usable_ids = non_pad if int(non_pad.shape[0]) > 0 else generated_ids
            response = self.tokenizer.decode(usable_ids, skip_special_tokens=True).strip()
            token_count = int(usable_ids.shape[0])
            results.append((response, max(1, token_count)))
        return results
