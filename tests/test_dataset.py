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
from typing import Any

import numpy as np
import pytest
import torch

from glp.dataset import (
    BuildConfig,
    DatasetConfig,
    ExtractConfig,
    FilterConfig,
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


# ── token-budget stop (extract.max_tokens) ────────────────────────────────────


def test_max_tokens_stops_extraction_early(tmp_path: Path) -> None:
    # 20 prompts x seq_len 3 (all-ones mask) = 60 token-activations available.
    data = np.random.default_rng(5).standard_normal((20, 8)).astype(np.float32)
    cfg = make_cfg(tmp_path, granularity=("all",), file_size=4096)
    cfg.extract.max_tokens = 25  # num_gpus=1 -> per-shard budget 25

    build_shard(
        cfg,
        0,
        backend=FakeBackend(make_batches(torch.from_numpy(data), batch_size=4)),
        texts=["x"] * 20,
    )
    finalize(cfg)

    n = len(load_activation_dataset(str(Path(cfg.output_dir) / "all" / "layer_07")))
    # stopped at the first batch that crossed the budget: 3 batches x 12 tokens = 36,
    # i.e. >= budget but overshooting by < one batch (12), and well short of all 60.
    assert 25 <= n <= 25 + 12
    assert n < 60


def test_max_tokens_none_collects_everything(tmp_path: Path) -> None:
    data = np.random.default_rng(6).standard_normal((20, 8)).astype(np.float32)
    cfg = make_cfg(tmp_path, granularity=("all",), file_size=4096)
    assert cfg.extract.max_tokens is None  # default: no cap

    build_shard(
        cfg,
        0,
        backend=FakeBackend(make_batches(torch.from_numpy(data), batch_size=4)),
        texts=["x"] * 20,
    )
    finalize(cfg)

    n = len(load_activation_dataset(str(Path(cfg.output_dir) / "all" / "layer_07")))
    assert n == 60  # 20 prompts x 3 tokens each


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


def test_resolve_add_special_tokens() -> None:
    from glp.dataset.backends import resolve_add_special_tokens

    def cfg(fmt: str, override: bool | None = None) -> BuildConfig:
        return BuildConfig(
            model_name="m",
            output_dir="o",
            dataset=DatasetConfig(path="p", format=fmt),
            extract=ExtractConfig(layers=[0], add_special_tokens=override),
        )

    # chat-templated text already carries the template's specials -> don't re-add BOS
    assert resolve_add_special_tokens(cfg("chat")) is False
    # plain text -> the tokenizer must add BOS
    assert resolve_add_special_tokens(cfg("text")) is True
    # an explicit config value overrides the format-based default
    assert resolve_add_special_tokens(cfg("chat", override=True)) is True
    assert resolve_add_special_tokens(cfg("text", override=False)) is False


# ── layer-spec resolution / config parsing ───────────────────────────────────


def test_resolve_layers_all_and_list() -> None:
    from glp.dataset.builder import resolve_layer_spec

    assert resolve_layer_spec("all", 16) == list(range(16))
    assert resolve_layer_spec([8, 12, 14], 0) == [8, 12, 14]
    with pytest.raises(ValueError):
        resolve_layer_spec("everything", 16)
    with pytest.raises(ValueError):
        resolve_layer_spec("all", 0)


def test_from_dict_helpers_match_build_config() -> None:
    from glp.dataset import dataset_config_from_dict, extract_config_from_dict

    data: dict[str, Any] = {
        "model_name": "fake-model",
        "output_dir": "out",
        "backend": "hf_baukit",
        "num_gpus": 2,
        "dataset": {
            "path": "fake-dataset",
            "format": "chat",
            "prompt_view": "user",
            "dedup": True,
            "filters": [{"column": "lang", "equals": "en"}],
        },
        "extract": {
            "layers": [8, 12, 14],
            "granularity": ["last", "all"],
            "dtype": "bfloat16",
            "max_tokens": 100,
        },
    }
    cfg = BuildConfig.from_dict(data)
    # the factored helpers are exactly what BuildConfig.from_dict uses
    assert cfg.dataset == dataset_config_from_dict(dict(data["dataset"]))
    assert cfg.extract == extract_config_from_dict(dict(data["extract"]), "fake-model")
    assert cfg.extract.layers == [8, 12, 14]
    assert cfg.extract.max_tokens == 100
    assert cfg.dataset.filters == [FilterConfig(column="lang", equals="en")]


def test_filter_config_parses_isin_and_rejects_ambiguity() -> None:
    from glp.dataset import dataset_config_from_dict

    cfg = dataset_config_from_dict(
        {
            "path": "fake-dataset",
            "split": "train",
            "filters": [{"column": "origin", "isin": ["wildchat_4m", "lmsys"]}],
        }
    )
    assert cfg.split == "train"
    assert cfg.filters == [FilterConfig(column="origin", isin=["wildchat_4m", "lmsys"])]
    assert cfg.filters[0].allowed == ["wildchat_4m", "lmsys"]
    assert FilterConfig(column="origin", equals="wildchat_4m").allowed == [
        "wildchat_4m"
    ]

    for bad in ({}, {"equals": "a", "isin": ["a"]}):
        with pytest.raises(ValueError, match="exactly one"):
            FilterConfig(column="origin", **bad)  # type: ignore[arg-type]


# ── stats pre-pass (stacked multi-layer statistics) ───────────────────────────


def make_multilayer_batches(
    data_by_layer: torch.Tensor, batch_size: int
) -> list[BatchActs]:
    """Build ``(B, L, S=1, D)`` batches from per-layer data ``(L, N, D)``.

    With an all-ones mask and seq_len 1, both ``last`` and ``all`` pooling
    recover every sample exactly, one per (prompt, layer).
    """
    _, n, _ = data_by_layer.shape
    batches: list[BatchActs] = []
    for start in range(0, n, batch_size):
        chunk = data_by_layer[:, start : start + batch_size, :]  # (L, B, D)
        acts = chunk.permute(1, 0, 2)[:, :, None, :]  # (B, L, 1, D)
        mask = torch.ones(chunk.shape[1], 1, dtype=torch.long)
        batches.append((acts.contiguous(), mask))
    return batches


def test_compute_layer_stats_matches_numpy(tmp_path: Path) -> None:
    from glp.dataset.builder import compute_layer_stats

    rng = np.random.default_rng(7)
    data = torch.from_numpy(rng.standard_normal((2, 40, 8)).astype(np.float32))
    cfg = make_cfg(tmp_path, granularity=("all",), layers=(3, 9))

    stats, tokens = compute_layer_stats(
        cfg,
        backend=FakeBackend(make_multilayer_batches(data, batch_size=8)),
        texts=["x"] * 40,
    )
    assert sorted(stats) == [3, 9]
    assert tokens == 40  # seq_len 1, all-ones mask
    for row, layer in enumerate([3, 9]):
        assert np.allclose(stats[layer].mean, data[row].numpy().mean(axis=0), atol=1e-5)
        assert np.allclose(stats[layer].var, data[row].numpy().var(axis=0), atol=1e-5)


def test_compute_layer_stats_max_tokens_cap(tmp_path: Path) -> None:
    from glp.dataset.builder import compute_layer_stats

    data = torch.randn(1, 40, 8)
    cfg = make_cfg(tmp_path, granularity=("all",), layers=(0,))
    cfg.extract.max_tokens = 10

    stats, tokens = compute_layer_stats(
        cfg,
        backend=FakeBackend(make_multilayer_batches(data, batch_size=8)),
        texts=["x"] * 40,
    )
    # stops at the first batch crossing the budget (whole batches of 8 tokens)
    assert 10 <= tokens <= 16
    assert stats[0].count == tokens


def test_compute_layer_stats_requires_single_granularity(tmp_path: Path) -> None:
    from glp.dataset.builder import compute_layer_stats

    cfg = make_cfg(tmp_path, granularity=("last", "all"))
    with pytest.raises(ValueError, match="exactly one granularity"):
        compute_layer_stats(cfg, backend=FakeBackend([]), texts=["x"])


def test_stacked_stats_roundtrip_through_normalizer(tmp_path: Path) -> None:
    from glp.dataset.builder import compute_layer_stats
    from glp.dataset.stats import stacked_normalizer_tensors

    rng = np.random.default_rng(8)
    data = torch.from_numpy(rng.standard_normal((2, 40, 8)).astype(np.float32))
    data[1] = data[1] * 3.0 + 5.0  # give layer 9 a distinct scale/offset
    cfg = make_cfg(tmp_path, granularity=("all",), layers=(3, 9))
    stats, _ = compute_layer_stats(
        cfg,
        backend=FakeBackend(make_multilayer_batches(data, batch_size=8)),
        texts=["x"] * 40,
    )

    mean, var = stacked_normalizer_tensors(stats, n_layers_total=16)
    assert mean.shape == var.shape == (16, 8)
    # unmeasured layers are NaN so misuse fails loudly
    assert torch.isnan(mean[0]).all() and torch.isnan(var[15]).all()
    assert torch.isfinite(mean[3]).all() and torch.isfinite(var[9]).all()

    stats_path = tmp_path / "rep_statistics.pt"
    torch.save({"mean": mean, "var": var}, stats_path)
    norm = Normalizer.from_config(stats_path)

    # per-sample layer_idx picks each sample's own layer stats
    latents = torch.stack([data[0, 0][None, :], data[1, 0][None, :]])  # (2, 1, 8)
    layer_idx = torch.tensor([3, 9])
    normalized = norm.normalize(latents, layer_idx=layer_idx)
    expected0 = (data[0, 0] - mean[3]) / torch.sqrt(var[3])
    expected9 = (data[1, 0] - mean[9]) / torch.sqrt(var[9])
    assert torch.allclose(normalized[0, 0], expected0, atol=1e-5)
    assert torch.allclose(normalized[1, 0], expected9, atol=1e-5)
    # round-trip
    restored = norm.denormalize(normalized, layer_idx=layer_idx)
    assert torch.allclose(restored, latents, atol=1e-4)


def test_stack_rep_statistics_from_layer_dirs(tmp_path: Path) -> None:
    from glp.dataset.stats import stack_rep_statistics

    for layer, offset in [(8, 1.0), (12, 2.0)]:
        layer_dir = tmp_path / f"layer_{layer:02d}"
        layer_dir.mkdir()
        torch.save(
            {"mean": torch.full((4,), offset), "var": torch.full((4,), offset * 2)},
            layer_dir / "rep_statistics.pt",
        )

    out = tmp_path / "stacked" / "rep_statistics.pt"
    stack_rep_statistics(
        [tmp_path / "layer_08", tmp_path / "layer_12"], n_layers_total=16, out_path=out
    )
    payload = torch.load(out)
    assert payload["mean"].shape == (16, 4)
    assert torch.allclose(payload["mean"][8], torch.full((4,), 1.0))
    assert torch.allclose(payload["var"][12], torch.full((4,), 4.0))
    assert torch.isnan(payload["mean"][0]).all()


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
