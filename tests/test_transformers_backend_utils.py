from __future__ import annotations

from types import SimpleNamespace

from art.backends.transformers_backend import (
    _build_generation_kwargs,
    _coerce_token_id,
    _resolve_eos_token_id,
)


def test_coerce_token_id_handles_scalars_and_nested_sequences() -> None:
    assert _coerce_token_id(None) is None
    assert _coerce_token_id(7) == 7
    assert _coerce_token_id("9") == 9
    assert _coerce_token_id([None, "11"]) == 11
    assert _coerce_token_id((None, ["13"])) == 13
    assert _coerce_token_id(["bad", None]) is None


def test_resolve_eos_token_id_falls_back_to_model_config() -> None:
    tokenizer = SimpleNamespace(eos_token_id=None)
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[None, 42]))
    assert _resolve_eos_token_id(tokenizer, model) == 42


def test_build_generation_kwargs_omits_missing_ids() -> None:
    kwargs = _build_generation_kwargs(
        max_new_tokens=128,
        temperature=0.0,
        pad_token_id=None,
        eos_token_id=None,
    )
    assert kwargs["max_new_tokens"] == 128
    assert kwargs["do_sample"] is False
    assert "pad_token_id" not in kwargs
    assert "eos_token_id" not in kwargs


def test_build_generation_kwargs_includes_ids_and_sampling() -> None:
    kwargs = _build_generation_kwargs(
        max_new_tokens=64,
        temperature=0.7,
        pad_token_id=3,
        eos_token_id=2,
    )
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["pad_token_id"] == 3
    assert kwargs["eos_token_id"] == 2
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.7
