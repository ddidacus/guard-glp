"""CPU-only tests for the streaming (on-the-fly) activation pipeline.

Everything runs hermetically with injected fake backends and in-process
``queue.Queue`` transport (the ``ChunkSource`` seam is identical for the
``torch.multiprocessing`` per-rank queues used in production). Covers the
producer -> consumer round-trip (exactly-once delivery in single-pass mode),
shuffle-buffer determinism and layer mixing, the val broadcast, cycling,
error propagation, and ``StreamingConfig`` parsing.
"""

import queue
import threading
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from glp.dataset import (
    DatasetConfig,
    ExtractConfig,
    QueueChunkSource,
    StreamingActDataset,
    StreamingConfig,
    start_thread_producer,
)
from glp.dataset.backends import BatchActs, ExtractionBackend

DIM = 4


class IndexedBackend:
    """Deterministic fake backend: sample value encodes (text index, layer).

    Texts must be stringified integers. Every text becomes one token whose
    activation at layer ``l`` is the constant vector ``idx + 1000 * l``, so each
    delivered sample is attributable to exactly one (text, layer) pair.
    """

    def __init__(self, layers: list[int], batch_size: int = 4) -> None:
        self.layers = layers
        self.batch_size = batch_size

    def iter_batches(self, texts: list[str]) -> Iterator[BatchActs]:
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            acts = torch.zeros(len(batch), len(self.layers), 1, DIM)
            for b, text in enumerate(batch):
                for layer_pos, layer in enumerate(self.layers):
                    acts[b, layer_pos, 0, :] = int(text) + 1000.0 * layer
            mask = torch.ones(len(batch), 1, dtype=torch.long)
            yield acts, mask


class FailingBackend:
    def iter_batches(self, texts: list[str]) -> Iterator[BatchActs]:
        raise RuntimeError("backend exploded")
        yield  # pragma: no cover


def make_streaming_cfg(layers: list[int], **overrides: object) -> StreamingConfig:
    defaults: dict[str, object] = {
        "dataset": DatasetConfig(path="fake-dataset"),
        "extract": ExtractConfig(layers=layers, granularity=["all"], dtype="float32"),
        "backend": "hf_baukit",
        "num_producers": 0,
        "chunk_samples": 8,
        "queue_maxsize": 4,
        "shuffle_buffer_size": 32,
        "shuffle_min_fill": 0.5,
        "cycle": False,
        "stall_timeout_s": 10.0,
    }
    defaults.update(overrides)
    return StreamingConfig(**defaults)  # type: ignore[arg-type]


def run_producer_consumer(
    cfg: StreamingConfig,
    n_texts: int,
    seed: int = 0,
    max_samples: int | None = None,
    backend: ExtractionBackend | None = None,
) -> tuple[list[dict[str, torch.Tensor]], StreamingActDataset, threading.Event]:
    """Start a thread producer over ``n_texts`` and consume the stream."""
    q: queue.Queue[Any] = queue.Queue(maxsize=cfg.queue_maxsize)
    stop = threading.Event()
    layers = list(cfg.extract.layers)
    start_thread_producer(
        cfg,
        "fake-model",
        q,
        stop,
        backend=backend if backend is not None else IndexedBackend(layers),
        texts=[str(i) for i in range(n_texts)],
        seed=seed,
    )
    dataset = StreamingActDataset(
        source=QueueChunkSource(q),
        num_producers=1,
        buffer_size=cfg.shuffle_buffer_size,
        min_fill=cfg.shuffle_min_fill,
        seed=seed,
        stall_timeout_s=cfg.stall_timeout_s,
    )
    samples = []
    for sample in dataset:
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples, dataset, stop


def sample_ids(samples: list[dict[str, torch.Tensor]]) -> list[int]:
    """Recover the (text idx + 1000*layer) identifier of each sample."""
    return [int(s["activations"][0, 0].item()) for s in samples]


# ── round-trip: exactly-once delivery, ActDataset-shaped samples ──────────────


def test_single_pass_delivers_every_sample_exactly_once() -> None:
    layers = [3, 9]
    cfg = make_streaming_cfg(layers)
    n_texts = 40
    samples, _, _ = run_producer_consumer(cfg, n_texts)

    assert len(samples) == n_texts * len(layers)
    expected = {i + 1000 * layer for i in range(n_texts) for layer in layers}
    assert set(sample_ids(samples)) == expected  # every sample exactly once

    sample = samples[0]
    assert sample["activations"].shape == (1, DIM)
    assert sample["activations"].dtype == torch.float32
    assert sample["layer_idx"].dtype == torch.long
    # layer_idx tags match the value encoding
    for s in samples:
        ident = int(s["activations"][0, 0].item())
        assert int(s["layer_idx"].item()) == ident // 1000


def test_shuffle_is_deterministic_per_seed_and_mixes_layers() -> None:
    layers = [0, 1]
    cfg = make_streaming_cfg(layers)
    ids_a = sample_ids(run_producer_consumer(cfg, 64, seed=7)[0])
    ids_b = sample_ids(run_producer_consumer(cfg, 64, seed=7)[0])
    ids_c = sample_ids(run_producer_consumer(cfg, 64, seed=8)[0])
    assert ids_a == ids_b  # same seed -> identical stream order
    assert ids_a != ids_c  # different seed -> different order
    assert ids_a != sorted(ids_a)  # actually shuffled

    # both layers appear within a small window once yielding starts
    window = [i // 1000 for i in ids_a[:32]]
    assert set(window) == {0, 1}


def test_bfloat16_wire_dtype_upcasts_at_yield() -> None:
    cfg = make_streaming_cfg([0])
    cfg.extract.dtype = "bfloat16"
    samples, _, _ = run_producer_consumer(cfg, 16)
    assert all(s["activations"].dtype == torch.float32 for s in samples)


# ── val broadcast ─────────────────────────────────────────────────────────────


def test_val_broadcast_arrives_before_training_stream() -> None:
    layers = [2, 5]
    cfg = make_streaming_cfg(layers, val_num_prompts=4)
    q: queue.Queue[Any] = queue.Queue(maxsize=cfg.queue_maxsize)
    stop = threading.Event()
    start_thread_producer(
        cfg,
        "fake-model",
        q,
        stop,
        backend=IndexedBackend(layers),
        texts=[str(i) for i in range(24)],
    )
    dataset = StreamingActDataset(
        source=QueueChunkSource(q),
        num_producers=1,
        buffer_size=cfg.shuffle_buffer_size,
        min_fill=cfg.shuffle_min_fill,
        seed=0,
        stall_timeout_s=cfg.stall_timeout_s,
    )
    val = dataset.wait_for_val()
    # texts 0..3 are the held-out prompts, one sample per (prompt, layer)
    assert set(sample_ids(val)) == {
        i + 1000 * layer for i in range(4) for layer in layers
    }

    train = list(dataset)
    assert set(sample_ids(train)) == {
        i + 1000 * layer for i in range(4, 24) for layer in layers
    }
    # val and train are disjoint
    assert not set(sample_ids(val)) & set(sample_ids(train))


def test_val_samples_per_layer_cap() -> None:
    layers = [0]
    cfg = make_streaming_cfg(layers, val_num_prompts=16, val_samples_per_layer=5)
    q: queue.Queue[Any] = queue.Queue(maxsize=64)
    stop = threading.Event()
    start_thread_producer(
        cfg,
        "fake-model",
        q,
        stop,
        backend=IndexedBackend(layers),
        texts=[str(i) for i in range(32)],
    )
    dataset = StreamingActDataset(
        source=QueueChunkSource(q),
        num_producers=1,
        buffer_size=cfg.shuffle_buffer_size,
        min_fill=cfg.shuffle_min_fill,
        seed=0,
        stall_timeout_s=cfg.stall_timeout_s,
    )
    assert len(dataset.wait_for_val()) == 5


# ── cycling and shutdown ──────────────────────────────────────────────────────


def test_cycle_reiterates_corpus_until_stopped() -> None:
    layers = [0]
    cfg = make_streaming_cfg(layers, cycle=True)
    n_texts = 16  # 16 samples per pass; ask for far more than one pass
    samples, _, stop = run_producer_consumer(cfg, n_texts, max_samples=100)
    assert len(samples) == 100
    ids = sample_ids(samples)
    assert set(ids) == set(range(16))  # only corpus samples, repeated
    stop.set()  # release the producer thread blocked on a full queue


def test_stall_timeout_raises() -> None:
    q: queue.Queue[Any] = queue.Queue()
    dataset = StreamingActDataset(
        source=QueueChunkSource(q),
        num_producers=1,
        buffer_size=8,
        min_fill=0.5,
        seed=0,
        stall_timeout_s=0.1,
    )
    with pytest.raises(RuntimeError, match="stalled"):
        next(iter(dataset))


# ── error propagation ─────────────────────────────────────────────────────────


def test_producer_error_propagates_to_consumer() -> None:
    cfg = make_streaming_cfg([0])
    with pytest.raises(RuntimeError, match="backend exploded"):
        run_producer_consumer(cfg, 8, backend=FailingBackend())


# ── config parsing ────────────────────────────────────────────────────────────


def test_streaming_config_from_dict_defaults_and_validation() -> None:
    data = {
        "dataset": {"path": "fake-dataset", "format": "chat", "prompt_view": "user"},
        "extract": {"layers": [8, 12], "granularity": ["all"], "dtype": "bfloat16"},
        "backend": "vllm_nnsight",
        "num_producers": 2,
        "val_num_prompts": 64,
    }
    cfg = StreamingConfig.from_dict(data, "fake-model")
    assert cfg.extract.layers == [8, 12]
    assert cfg.backend == "vllm_nnsight"
    assert cfg.num_producers == 2
    assert cfg.val_num_prompts == 64
    assert cfg.cycle is True  # default
    assert cfg.chunk_samples == 8192  # default
    assert cfg.granularity == "all"

    with pytest.raises(ValueError, match="exactly one granularity"):
        make_streaming_cfg(
            [0], extract=ExtractConfig(layers=[0], granularity=["last", "all"])
        )


def test_multi_producer_slices_partition_corpus() -> None:
    """Two producers on disjoint slices together deliver the whole corpus."""
    layers = [1]
    cfg = make_streaming_cfg(layers, num_producers=2)
    q: queue.Queue[Any] = queue.Queue(maxsize=64)
    stop = threading.Event()
    texts = [str(i) for i in range(30)]
    threads = []
    for producer_id in range(2):
        from glp.dataset.streaming import producer_loop

        t = threading.Thread(
            target=producer_loop,
            args=(cfg, "fake-model", producer_id, [q], stop),
            kwargs={"backend": IndexedBackend(layers), "texts": texts},
            daemon=True,
        )
        t.start()
        threads.append(t)
    dataset = StreamingActDataset(
        source=QueueChunkSource(q),
        num_producers=2,
        buffer_size=cfg.shuffle_buffer_size,
        min_fill=cfg.shuffle_min_fill,
        seed=0,
        stall_timeout_s=cfg.stall_timeout_s,
    )
    samples = list(dataset)
    assert set(sample_ids(samples)) == {i + 1000 for i in range(30)}
    assert len(samples) == 30
