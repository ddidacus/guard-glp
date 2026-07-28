"""Tests for the shared activation cache (no model / no network)."""

from pathlib import Path
from typing import Any

import torch

from glp.dataset.act_cache import cache_key, cached_activations


def test_cache_key_encodes_determining_fields() -> None:
    k = cache_key(
        dataset="wildjailbreak_vanilla",
        llm_model_id="meta-llama/Llama-3.2-1B-Instruct",
        layers=[14],
        token_pooling="mean",
        split="test_good",
        shard=0,
    )
    assert k == (
        "wildjailbreak_vanilla__meta-llama-Llama-3.2-1B-Instruct__L14__mean__"
        "test_good__s0.th"
    )


def test_key_differs_on_llm_layer_and_pooling() -> None:
    def key(
        llm: str = "meta-llama/Llama-3.2-1B-Instruct",
        layers: list[int] | None = None,
        pooling: str = "mean",
    ) -> str:
        return cache_key(
            dataset="d",
            llm_model_id=llm,
            layers=layers or [14],
            token_pooling=pooling,
            split="test_good",
            shard=0,
        )

    k0 = key()
    assert k0 != key(llm="meta-llama/Llama-3.2-1B")
    assert k0 != key(layers=[7])
    assert k0 != key(pooling="last")


def test_extract_called_once_then_cached(tmp_path: Path) -> None:
    calls = {"n": 0}

    def _extract() -> torch.Tensor:
        calls["n"] += 1
        return torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)

    kw: dict[str, Any] = {
        "dataset": "d",
        "llm_model_id": "m",
        "layers": [14],
        "token_pooling": "mean",
        "split": "test_good",
        "shard": 0,
        "cache_dir": tmp_path,
    }
    a = cached_activations(extract=_extract, **kw)
    b = cached_activations(extract=_extract, **kw)
    assert calls["n"] == 1  # second call is a cache hit
    assert torch.equal(a, b)


def test_different_key_triggers_new_extract(tmp_path: Path) -> None:
    calls = {"n": 0}

    def _extract() -> torch.Tensor:
        calls["n"] += 1
        return torch.zeros(1, 1, 3)

    common: dict[str, Any] = {
        "dataset": "d",
        "llm_model_id": "m",
        "layers": [14],
        "token_pooling": "mean",
        "shard": 0,
        "cache_dir": tmp_path,
        "extract": _extract,
    }
    cached_activations(split="test_good", **common)
    cached_activations(split="test_bad", **common)
    assert calls["n"] == 2  # different split -> different key -> re-extract
