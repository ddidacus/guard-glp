"""CPU-only tests for streaming (on-the-fly) GLP training.

Runs the full streaming path hermetically: a producer thread with an injected
deterministic backend feeds a chunk queue (exactly the launcher's per-rank
transport, minus multiprocessing), the trainer consumes it with the shuffle
buffer, normalizes with STACKED per-layer stats, conditions the denoiser on
``layer_idx`` (``multi_layer_n_layers``), evaluates on the broadcast val set,
and checkpoints. Also covers the gloo single-rank DDP wrap and the fail-fast
config/stats validations.
"""

import queue
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from glp.dataset.backends import BatchActs
from glp.dataset.streaming import StreamingConfig, start_thread_producer
from glp.train import train

DIM = 16
LAYERS = [3, 9]
N_TEXTS = 96
N_LAYERS_TOTAL = 16


def _vec(idx: int, layer: int) -> torch.Tensor:
    """Deterministic per-(text, layer) activation with per-layer offset/scale."""
    g = torch.Generator().manual_seed(idx * 1009 + layer * 7919)
    return torch.randn(DIM, generator=g) * (1.0 + 0.1 * layer) + layer


class DeterministicBackend:
    """One token per text; activation determined by (text index, layer)."""

    def __init__(self, batch_size: int = 8) -> None:
        self.batch_size = batch_size

    def iter_batches(self, texts: list[str]) -> Iterator[BatchActs]:
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            acts = torch.stack(
                [
                    torch.stack([_vec(int(t), layer) for layer in LAYERS])[:, None, :]
                    for t in batch
                ]
            )  # (B, L, 1, D)
            mask = torch.ones(len(batch), 1, dtype=torch.long)
            yield acts, mask


def _write_stacked_stats(path: Path, layers: list[int] = LAYERS) -> None:
    """Exact per-layer stats over the full text population; NaN for other rows."""
    mean = torch.full((N_LAYERS_TOTAL, DIM), float("nan"))
    var = torch.full((N_LAYERS_TOTAL, DIM), float("nan"))
    for layer in layers:
        pool = torch.stack([_vec(i, layer) for i in range(N_TEXTS)])
        mean[layer] = pool.mean(dim=0)
        var[layer] = pool.var(dim=0, unbiased=False)
    torch.save({"mean": mean, "var": var}, path)


def _make_config(tmp_path: Path, **overrides: Any) -> DictConfig:
    rep = str(tmp_path / "rep_statistics.pt")
    base: dict[str, Any] = {
        "model_name": "fake-model",
        "output_path": str(tmp_path / "run"),
        "rep_statistic": rep,
        "data_mode": "streaming",
        "streaming": {
            "backend": "hf_baukit",
            "num_producers": 1,  # fed via an injected chunk_queue in these tests
            "dataset": {"path": "fake-dataset"},
            "extract": {
                "layers": LAYERS,
                "granularity": ["all"],
                "dtype": "float32",
            },
            "chunk_samples": 16,
            "queue_maxsize": 8,
            "shuffle_buffer_size": 64,
            "shuffle_min_fill": 0.5,
            "val_num_prompts": 8,
            "cycle": True,
            "stall_timeout_s": 30.0,
        },
        "epoch_size": 256,  # samples per rank -> 8 steps at batch_size 32
        "num_epochs": 1,
        "val_every_n_steps": 4,
        "val_max_batches": 2,
        "use_bf16": False,  # CPU
        "batch_size": 32,
        "learning_rate": 1e-3,
        "log_every_n_steps": 2,
        "save_epochs": [1],
        "save_opt_state": True,
        "seed": 0,
        "lr_scheduler": {
            "scheduler_cls": "cosine_scheduler_with_warmup",
            "warmup_ratio": 0.1,
            "initial_factor": 0.01,
            "final_factor": 0.1,
        },
        "glp_kwargs": {
            "normalizer_config": {"rep_statistic": rep},
            "denoiser_config": {
                "d_input": DIM,
                "d_model": 2 * DIM,
                "d_mlp": 4 * DIM,
                "n_layers": 2,
                "multi_layer_n_layers": N_LAYERS_TOTAL,  # layer conditioning ON
            },
            "tracedict_config": {
                "layer_prefix": "model.layers",
                "layers": LAYERS,
                "retain": "output",
            },
        },
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _start_producer(
    config: DictConfig,
) -> tuple[queue.Queue[Any], threading.Event]:
    """Producer thread over the fake corpus, exactly like one launcher producer."""
    scfg = StreamingConfig.from_dict(
        OmegaConf.to_container(config.streaming, resolve=True),  # type: ignore[arg-type]
        config.model_name,
    )
    q: queue.Queue[Any] = queue.Queue(maxsize=scfg.queue_maxsize)
    stop = threading.Event()
    start_thread_producer(
        scfg,
        config.model_name,
        q,
        stop,
        backend=DeterministicBackend(),
        texts=[str(i) for i in range(N_TEXTS)],
        seed=0,
    )
    return q, stop


def test_streaming_train_smoke(tmp_path: Path) -> None:
    _write_stacked_stats(tmp_path / "rep_statistics.pt")
    config = _make_config(tmp_path)
    q, stop = _start_producer(config)
    try:
        model = train(config, device="cpu", chunk_queue=q, producer_stop=stop)
    finally:
        stop.set()

    out = tmp_path / "run"
    assert (out / "final.safetensors").exists()
    assert (out / "rep_statistics.pt").exists()
    assert (out / "config.yaml").exists()
    assert (out / "checkpoints" / "epoch_1.safetensors").exists()
    # the layer-conditioning embedding was actually built and trained
    assert any("layer_embed" in name for name, _ in model.named_parameters())
    # train() sets producer_stop on completion so producers exit
    assert stop.is_set()


def test_streaming_train_ddp_single_rank_gloo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WORLD_SIZE=1 exercises init_process_group + DDP wrap + all_reduce + barrier."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29517")

    _write_stacked_stats(tmp_path / "rep_statistics.pt")
    config = _make_config(tmp_path)
    q, stop = _start_producer(config)
    try:
        train(config, device="cpu", chunk_queue=q, producer_stop=stop)
    finally:
        stop.set()
        import torch.distributed as dist

        if dist.is_initialized():  # train() destroys the group on the clean path
            dist.destroy_process_group()
    assert (tmp_path / "run" / "final.safetensors").exists()


def test_streaming_requires_epoch_size(tmp_path: Path) -> None:
    _write_stacked_stats(tmp_path / "rep_statistics.pt")
    config = _make_config(tmp_path, epoch_size=None)
    q: queue.Queue[Any] = queue.Queue()
    with pytest.raises(ValueError, match="epoch_size"):
        train(config, device="cpu", chunk_queue=q)


def test_streaming_nan_stats_fail_fast(tmp_path: Path) -> None:
    # stats only cover layer 3; streaming layers [3, 9] must be rejected
    _write_stacked_stats(tmp_path / "rep_statistics.pt", layers=[3])
    config = _make_config(tmp_path)
    q, stop = _start_producer(config)
    try:
        with pytest.raises(ValueError, match="non-finite stats for layer 9"):
            train(config, device="cpu", chunk_queue=q, producer_stop=stop)
    finally:
        stop.set()


def test_static_mode_rejects_distributed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    _write_stacked_stats(tmp_path / "rep_statistics.pt")
    config = _make_config(tmp_path, data_mode="static")
    with pytest.raises(ValueError, match="static"):
        train(config, device="cpu")
