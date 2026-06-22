"""CPU-only tests for the dataset manager (no network, no GPU by default).

Covers ``RunningStats`` correctness and shard-merge equality against numpy
single-pass, the ``build_shard`` -> ``finalize`` -> ``load_activation_dataset``
round-trip (float32 and the bfloat16/int16 encoding), multi-shard merging, and
the pooling granularity sample counts. The hermetic tests use an injected fake
backend so they need no model download; a ``slow`` test exercises the real
``HFBaukitBackend`` against a tiny HF model when network is available.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from glp.dataset import (
    BuildConfig,
    DatasetConfig,
    ExtractConfig,
    HFBaukitBackend,
    RunningStats,
    build_shard,
    finalize,
    get_activation_dataloader,
    load_activation_dataset,
)
from glp.dataset.backends import BatchActs
from glp.denoiser import Normalizer
from glp.utils_acts import pool_activations

# ── fixtures / helpers ────────────────────────────────────────────────────────


class FakeBackend:
    """Yields pre-built ``(acts, mask)`` batches, bypassing any model."""

    def __init__(self, batches: list[BatchActs]) -> None:
        self.batches = batches

    def iter_batches(self, texts: list[str]) -> Iterator[BatchActs]:
        yield from self.batches


def make_batches(
    data: torch.Tensor, batch_size: int, seq_len: int = 3
) -> list[BatchActs]:
    """Build single-layer ``(B, 1, S, D)`` batches whose last token is ``data``.

    With a right-padded all-ones mask, ``last`` pooling recovers ``data`` exactly,
    so downstream stats and stored samples are predictable.
    """
    batches: list[BatchActs] = []
    n, dim = data.shape
    for start in range(0, n, batch_size):
        chunk = data[start : start + batch_size]  # (B, D)
        b = chunk.shape[0]
        acts = torch.zeros(b, 1, seq_len, dim, dtype=torch.float32)
        acts[:, 0, seq_len - 1, :] = chunk
        mask = torch.ones(b, seq_len, dtype=torch.long)
        batches.append((acts, mask))
    return batches


def make_cfg(
    tmp_path: Path,
    *,
    dtype: str = "float32",
    file_size: int = 24,
    granularity: tuple[str, ...] = ("last",),
    layers: tuple[int, ...] = (7,),
) -> BuildConfig:
    return BuildConfig(
        model_name="fake-model",
        output_dir=str(tmp_path / "out"),
        dataset=DatasetConfig(path="fake-dataset"),
        extract=ExtractConfig(
            layers=list(layers),
            granularity=list(granularity),
            dtype=dtype,
            file_size=file_size,
            batch_size=4,
            padding_side="right",
        ),
    )


# ── RunningStats ──────────────────────────────────────────────────────────────


def test_running_stats_matches_numpy() -> None:
    rng = np.random.default_rng(2)
    data = rng.standard_normal((100, 5))
    stats = RunningStats.zeros(5)
    for start in range(0, 100, 7):
        stats.update(torch.from_numpy(data[start : start + 7]))
    assert np.allclose(stats.mean, data.mean(axis=0), atol=1e-9)
    assert np.allclose(stats.var, data.var(axis=0), atol=1e-9)
    assert stats.count == 100


def test_running_stats_merge_equals_single_pass() -> None:
    rng = np.random.default_rng(3)
    data = rng.standard_normal((80, 5))
    left = RunningStats.zeros(5)
    right = RunningStats.zeros(5)
    left.update(torch.from_numpy(data[:33]))
    right.update(torch.from_numpy(data[33:]))
    left.merge(right)
    assert np.allclose(left.mean, data.mean(axis=0), atol=1e-9)
    assert np.allclose(left.var, data.var(axis=0), atol=1e-9)
    assert left.count == 80


def test_running_stats_dim_mismatch_raises() -> None:
    stats = RunningStats.zeros(4)
    with pytest.raises(ValueError):
        stats.update(torch.randn(3, 5))


# ── pooling granularity ───────────────────────────────────────────────────────


def test_granularity_sample_counts() -> None:
    acts = torch.randn(2, 1, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])  # lengths 3 and 4
    assert pool_activations(acts, mask, "last", "right").shape == (2, 1, 8)
    assert pool_activations(acts, mask, "mean", "right").shape == (2, 1, 8)
    # per-token: one sample per non-padding token (3 + 4 = 7)
    assert pool_activations(acts, mask, "all", "right").shape == (7, 1, 8)


# ── build -> finalize -> round-trip ───────────────────────────────────────────


def test_build_and_roundtrip_float32(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((20, 8)).astype(np.float32)
    cfg = make_cfg(tmp_path)

    build_shard(
        cfg,
        0,
        backend=FakeBackend(make_batches(torch.from_numpy(data), batch_size=4)),
        texts=["x"] * 20,
    )
    finalize(cfg)

    layer_dir = Path(cfg.output_dir) / "last" / "layer_07"
    assert (layer_dir / "dtype.txt").read_text().strip() == "float32"
    assert (layer_dir / "rep_statistics.pt").exists()

    manifest = json.loads((layer_dir / "manifest.json").read_text())
    assert manifest["num_samples"] == 20
    assert manifest["dim"] == 8
    assert manifest["granularity"] == "last"
    assert manifest["layer"] == 7

    # streamed+merged stats match numpy single-pass
    norm = Normalizer.from_config(layer_dir / "rep_statistics.pt")
    assert np.allclose(norm.mean.numpy(), data.mean(axis=0), atol=1e-5)
    assert np.allclose(norm.var.numpy(), data.var(axis=0), atol=1e-5)

    # round-trip through the in-repo loader
    dataset = load_activation_dataset(str(layer_dir))
    assert len(dataset) == 20
    sample = dataset[0]["activations"]
    assert sample.shape == (1, 8)
    assert np.allclose(sample.numpy()[0], data[0], atol=1e-5)

    loader = get_activation_dataloader(
        dataset, batch_size=4, normalizer=norm, shuffle=False
    )
    batch = next(iter(loader))
    assert batch["latents"].shape == (4, 1, 8)
    assert "layer_idx" in batch
    assert batch["layer_idx"].tolist() == [7, 7, 7, 7]


def test_build_and_roundtrip_bfloat16(tmp_path: Path) -> None:
    data = torch.randn(12, 8)
    cfg = make_cfg(tmp_path, dtype="bfloat16")

    build_shard(
        cfg,
        0,
        backend=FakeBackend(make_batches(data, batch_size=4)),
        texts=["x"] * 12,
    )
    finalize(cfg)

    layer_dir = Path(cfg.output_dir) / "last" / "layer_07"
    # bfloat16 is stored as int16 on disk and reinterpreted by ActDataset
    assert (layer_dir / "dtype.txt").read_text().strip() == "int16"

    dataset = load_activation_dataset(str(layer_dir))
    assert len(dataset) == 12
    sample = dataset[0]["activations"]
    assert sample.dtype == torch.float32
    expected = data[0].to(torch.bfloat16).float()
    assert torch.allclose(sample[0], expected, atol=1e-2)


def test_multishard_merge(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((30, 8)).astype(np.float32)
    cfg = make_cfg(tmp_path)

    build_shard(
        cfg,
        0,
        backend=FakeBackend(make_batches(torch.from_numpy(data[:18]), batch_size=4)),
        texts=["x"] * 18,
    )
    build_shard(
        cfg,
        1,
        backend=FakeBackend(make_batches(torch.from_numpy(data[18:]), batch_size=4)),
        texts=["x"] * 12,
    )
    finalize(cfg)

    layer_dir = Path(cfg.output_dir) / "last" / "layer_07"
    dataset = load_activation_dataset(str(layer_dir))
    assert len(dataset) == 30

    norm = Normalizer.from_config(layer_dir / "rep_statistics.pt")
    assert np.allclose(norm.mean.numpy(), data.mean(axis=0), atol=1e-5)
    assert np.allclose(norm.var.numpy(), data.var(axis=0), atol=1e-4)

    # all samples recoverable in (shard0, shard1) order; shard dirs cleaned up
    recovered = np.stack([dataset[i]["activations"].numpy()[0] for i in range(30)])
    assert np.allclose(recovered, data, atol=1e-5)
    assert not list(Path(cfg.output_dir).glob("shard_*"))


# ── real backend (network + tiny model) ───────────────────────────────────────


@pytest.mark.slow
def test_hf_baukit_backend_end_to_end(tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    cfg = make_cfg(tmp_path, layers=(0,), file_size=4096)
    cfg.extract.max_length = 32
    backend = HFBaukitBackend(
        model=model,
        tokenizer=tokenizer,
        tracedict_config={
            "layer_prefix": "model.layers",
            "layers": [0],
            "retain": "output",
        },
        batch_size=2,
        max_length=32,
        use_tqdm=False,
    )

    build_shard(cfg, 0, backend=backend, texts=["hello world", "the quick brown fox"])
    finalize(cfg)

    layer_dir = Path(cfg.output_dir) / "last" / "layer_00"
    dataset = load_activation_dataset(str(layer_dir))
    assert len(dataset) == 2
    norm = Normalizer.from_config(layer_dir / "rep_statistics.pt")
    loader = get_activation_dataloader(
        dataset, batch_size=2, normalizer=norm, shuffle=False
    )
    batch = next(iter(loader))
    assert batch["latents"].ndim == 3
    assert batch["latents"].shape[1] == 1
