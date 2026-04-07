from __future__ import annotations

import art.backends.factory as factory_module


def test_factory_routes_transformerlens(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _StubTLBackend:
        def __init__(self, *, model_id: str, tokenizer_id: str, device: str, dtype: str) -> None:
            captured["model_id"] = model_id
            captured["tokenizer_id"] = tokenizer_id
            captured["device"] = device
            captured["dtype"] = dtype

    monkeypatch.setattr(factory_module, "TransformerLensBackend", _StubTLBackend)

    backend = factory_module.create_backend(
        backend_name="transformerlens",
        model_id="gpt2",
        tokenizer_id="gpt2",
        device="cpu",
        dtype="float32",
    )

    assert isinstance(backend, _StubTLBackend)
    assert captured == {
        "model_id": "gpt2",
        "tokenizer_id": "gpt2",
        "device": "cpu",
        "dtype": "float32",
    }
