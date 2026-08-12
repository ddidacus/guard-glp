"""On-the-fly activation streaming: extraction producers feeding the trainer.

The static path extracts activations to disk (``builder``) and trains from
memmaps (``act_dataset``). This module is the streaming alternative: producer
processes (or an in-process thread) run an
:class:`~glp.dataset.backends.ExtractionBackend` over the corpus and push pooled
per-layer sample chunks through bounded queues straight into the trainer, which
shuffles them in a fixed-size in-memory buffer. Nothing touches disk; run length
is set by the trainer (``num_epochs * epoch_size``), and normalization stats come
from the stats pre-pass (``scripts/dataset/compute_stats.py``).

Wire protocol (``ChunkMsg``): tuples small enough to reason about and pickle-
friendly for ``torch.multiprocessing`` (CPU tensors travel via shared memory):

* ``("train", chunk (N, D), layer_id)`` — pooled samples from one LLM layer.
* ``("val", chunk (N, D), layer_id)`` — held-out validation samples, sent by
  producer 0 to every rank before any of its train chunks.
* ``("val_done", None, producer_id)`` — end of the validation broadcast.
* ``("end", None, producer_id)`` — this producer exhausted its corpus slice
  (only when ``cycle=False``).
* ``("error", None, message)`` — a producer failed; consumers re-raise.

Consumers see the same protocol whether the source is a ``queue.Queue`` fed by
an in-process thread (``num_producers=0``, the smoke/test path) or a per-rank
``torch.multiprocessing`` queue fed by producer processes — the
:class:`ChunkSource` protocol is the seam, and both queue types raise
``queue.Empty`` on timeout.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import torch
from torch.utils.data import IterableDataset

from glp.dataset.backends import ExtractionBackend, make_backend
from glp.dataset.builder import (
    BuildConfig,
    DatasetConfig,
    ExtractConfig,
    dataset_config_from_dict,
    extract_config_from_dict,
)
from glp.dataset.loader import load_texts
from glp.utils_acts import pool_activations

logger = logging.getLogger(__name__)

# ("train"|"val", chunk (N, D), layer_id) | ("val_done"|"end", None, producer_id)
# | ("error", None, message)
ChunkMsg = tuple[str, "torch.Tensor | None", "int | str"]

# Consumer poll interval: how long to wait for a chunk before falling back to
# draining the shuffle buffer. Short enough that a real producer hiccup is
# bridged from the buffer promptly, long enough that normal operation (producers
# keeping up) never drains — `get(timeout=...)` returns the instant a chunk
# arrives, so steady-state delivery is unaffected and stays deterministic.
_DRAIN_POLL_S = 5.0
_EMPTY = object()  # sentinel: no chunk available within the poll interval


# ── configuration ────────────────────────────────────────────────────────────


@dataclass
class StreamingConfig:
    """The ``streaming:`` section of a training config (``data_mode: streaming``)."""

    dataset: DatasetConfig
    extract: ExtractConfig
    backend: str = "hf_baukit"  # "hf_baukit" | "vllm_nnsight"
    # 0 = run the producer as a thread inside the trainer process (single GPU /
    # CPU smoke path); >= 1 = dedicated producer processes, one GPU each.
    num_producers: int = 1
    chunk_samples: int = 8192  # samples per queue message ((N, D) bf16 ~ 32 MB)
    queue_maxsize: int = 8  # per-rank bounded queue (backpressure / prefetch)
    shuffle_buffer_size: int = 500_000  # per-rank samples (~2 GB bf16 at D=2048)
    shuffle_min_fill: float = 0.5  # fill fraction required before yielding
    val_num_prompts: int = 0  # held-out prompts extracted once by producer 0
    val_samples_per_layer: int = 4096  # cap on retained val samples per layer
    cycle: bool = True  # producers loop their corpus slice (reshuffled per pass)
    stall_timeout_s: float = 600.0  # consumer get() timeout -> "producers stalled"
    # Budget for the FIRST chunk only. Producer startup is corpus-proportional (model
    # load, then load_texts: chat-templating + dedup over the whole corpus — ~6 min for
    # 3.1M prompts), which has nothing to do with steady-state liveness; sharing one
    # timeout for both means a bigger corpus trips the stall detector before training
    # ever starts. Keep stall_timeout_s tight so a genuinely dead producer is caught
    # fast, and give startup its own generous budget.
    startup_timeout_s: float = 3600.0

    def __post_init__(self) -> None:
        if len(self.extract.granularity) != 1:
            raise ValueError(
                "streaming requires exactly one granularity (got "
                f"{self.extract.granularity})"
            )

    @property
    def granularity(self) -> str:
        return self.extract.granularity[0]

    @classmethod
    def from_dict(cls, data: dict[str, Any], model_name: str) -> StreamingConfig:
        return cls(
            dataset=dataset_config_from_dict(dict(data["dataset"])),
            extract=extract_config_from_dict(dict(data["extract"]), model_name),
            backend=data.get("backend", "hf_baukit"),
            num_producers=int(data.get("num_producers", 1)),
            chunk_samples=int(data.get("chunk_samples", 8192)),
            queue_maxsize=int(data.get("queue_maxsize", 8)),
            shuffle_buffer_size=int(data.get("shuffle_buffer_size", 500_000)),
            shuffle_min_fill=float(data.get("shuffle_min_fill", 0.5)),
            val_num_prompts=int(data.get("val_num_prompts", 0)),
            val_samples_per_layer=int(data.get("val_samples_per_layer", 4096)),
            cycle=bool(data.get("cycle", True)),
            stall_timeout_s=float(data.get("stall_timeout_s", 600.0)),
            startup_timeout_s=float(data.get("startup_timeout_s", 3600.0)),
        )

    def to_build_config(self, model_name: str) -> BuildConfig:
        """Adapter so :func:`~glp.dataset.backends.make_backend` is reused as-is."""
        return BuildConfig(
            model_name=model_name,
            output_dir="",  # unused by make_backend
            dataset=self.dataset,
            extract=self.extract,
            backend=self.backend,
            num_gpus=max(self.num_producers, 1),
        )


# ── chunk sources (the consumer-side seam) ───────────────────────────────────


class ChunkSource(Protocol):
    """Blocking source of :data:`ChunkMsg` items (raises ``queue.Empty`` on timeout)."""

    def get(self, timeout: float | None = None) -> ChunkMsg: ...


class QueueChunkSource:
    """Wrap a ``queue.Queue`` or ``torch.multiprocessing`` queue as a ChunkSource."""

    def __init__(self, q: Any) -> None:
        self.q = q

    def get(self, timeout: float | None = None) -> ChunkMsg:
        return cast(ChunkMsg, self.q.get(timeout=timeout))


# ── producer ─────────────────────────────────────────────────────────────────


def _stream_dtype(config_dtype: str) -> torch.dtype:
    """Chunk dtype on the wire; mirrors the on-disk encoding choice."""
    return torch.bfloat16 if config_dtype == "bfloat16" else torch.float32


class _RoundRobinPutter:
    """Round-robin chunk delivery across rank queues, abortable via stop event."""

    def __init__(
        self,
        rank_queues: list[Any],
        start_offset: int,
        stop_event: Any | None,
    ) -> None:
        self.rank_queues = rank_queues
        self.next_rank = start_offset % len(rank_queues)
        self.stop_event = stop_event

    def stopped(self) -> bool:
        return self.stop_event is not None and self.stop_event.is_set()

    def put(self, msg: ChunkMsg, q: Any | None = None) -> bool:
        """Deliver a chunk; returns False if a stop was requested.

        ``q`` given (broadcast): blocking put to that one queue. ``q`` None
        (train stream): **fair** delivery — deliver to the first rank queue with
        room, scanning round-robin from ``next_rank``, and only wait when *every*
        queue is full (legitimate backpressure: trainers behind). This avoids
        head-of-line blocking, where a single full/slow rank queue would stall
        delivery to the other, hungry ranks (the failure mode that starved one
        rank for 600 s and tripped the NCCL watchdog).
        """
        if q is not None:
            while not self.stopped():
                try:
                    q.put(msg, timeout=1.0)
                except queue.Full:
                    continue
                return True
            return False

        n = len(self.rank_queues)
        while not self.stopped():
            for i in range(n):
                r = (self.next_rank + i) % n
                try:
                    self.rank_queues[r].put(msg, block=False)
                except queue.Full:
                    continue
                self.next_rank = (r + 1) % n
                return True
            # every rank queue is full -> trainers are behind; back off and retry
            time.sleep(0.05)
        return False

    def broadcast(self, msg: ChunkMsg) -> bool:
        return all(self.put(msg, q=q) for q in self.rank_queues)


def producer_loop(
    cfg: StreamingConfig,
    model_name: str,
    producer_id: int,
    rank_queues: list[Any],
    stop_event: Any | None = None,
    *,
    backend: ExtractionBackend | None = None,
    texts: list[str] | None = None,
    device: str | None = None,
    seed: int = 0,
) -> None:
    """Extract activations and stream pooled per-layer chunks to the rank queues.

    Producer ``producer_id`` handles the corpus slice
    ``texts[val_num_prompts:][producer_id::num_producers]``; producer 0
    additionally extracts the held-out ``texts[:val_num_prompts]`` first and
    broadcasts it to every rank (``"val"`` chunks then ``"val_done"``). With
    ``cycle=True`` the slice is re-iterated indefinitely (reshuffled each pass)
    until ``stop_event`` is set; with ``cycle=False`` one pass is made and
    ``("end", None, producer_id)`` is broadcast. ``backend``/``texts`` are
    injectable for tests, exactly like :func:`~glp.dataset.builder.build_shard`.
    On failure an ``("error", None, message)`` message is broadcast so consumers
    fail fast instead of hitting their stall timeout.
    """
    putter = _RoundRobinPutter(rank_queues, producer_id, stop_event)
    try:
        if backend is None:
            backend, tokenizer = make_backend(
                cfg.to_build_config(model_name), producer_id, device=device
            )
            if texts is None:
                # full capped corpus; producers slice it below (num_gpus=1 -> no stride)
                texts = load_texts(cfg.dataset, tokenizer, 0, 1)
        if texts is None:
            raise ValueError("texts must be provided when backend is injected")

        num_producers = max(cfg.num_producers, 1)
        val_texts = texts[: cfg.val_num_prompts]
        train_texts = texts[cfg.val_num_prompts :][producer_id::num_producers]
        if not train_texts:
            raise ValueError(
                f"producer {producer_id}: empty corpus slice "
                f"({len(texts)} texts, {cfg.val_num_prompts} val, "
                f"{num_producers} producers)"
            )

        if producer_id == 0 and val_texts:
            _broadcast_val(cfg, backend, val_texts, putter)

        granularity = cast(Literal["last", "mean", "all"], cfg.granularity)
        layers = list(cfg.extract.layers)
        dtype = _stream_dtype(cfg.extract.dtype)
        pending: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
        pending_rows: dict[int, int] = dict.fromkeys(layers, 0)

        pass_idx = 0
        while True:
            gen = torch.Generator().manual_seed(
                seed * 100_003 + producer_id * 1_009 + pass_idx
            )
            order = torch.randperm(len(train_texts), generator=gen).tolist()
            shuffled = [train_texts[i] for i in order]
            for acts, attention_mask in backend.iter_batches(shuffled):
                if putter.stopped():
                    return
                pooled = pool_activations(
                    acts, attention_mask, granularity, cfg.extract.padding_side
                )
                for layer_pos, layer in enumerate(layers):
                    samples = pooled[:, layer_pos, :].to(dtype)
                    pending[layer].append(samples)
                    pending_rows[layer] += samples.shape[0]
                    if pending_rows[layer] >= cfg.chunk_samples:
                        chunk = torch.cat(pending[layer], dim=0)
                        pending[layer] = []
                        pending_rows[layer] = 0
                        if not putter.put(("train", chunk, layer)):
                            return
            pass_idx += 1
            if not cfg.cycle:
                break
            logger.info(
                "producer %d: corpus pass %d complete; cycling", producer_id, pass_idx
            )

        # single-pass mode: flush partial chunks, then signal completion
        for layer in layers:
            if pending_rows[layer] > 0:
                chunk = torch.cat(pending[layer], dim=0)
                if not putter.put(("train", chunk, layer)):
                    return
        putter.broadcast(("end", None, producer_id))
        logger.info("producer %d: corpus exhausted; sent end", producer_id)
    except Exception as exc:  # noqa: BLE001 - forwarded to consumers
        logger.exception("producer %d failed", producer_id)
        putter.broadcast(("error", None, f"producer {producer_id}: {exc!r}"))
        raise


def _broadcast_val(
    cfg: StreamingConfig,
    backend: ExtractionBackend,
    val_texts: list[str],
    putter: _RoundRobinPutter,
) -> None:
    """Extract the held-out prompts once and broadcast them to every rank."""
    granularity = cast(Literal["last", "mean", "all"], cfg.granularity)
    layers = list(cfg.extract.layers)
    dtype = _stream_dtype(cfg.extract.dtype)
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    rows: dict[int, int] = dict.fromkeys(layers, 0)
    for acts, attention_mask in backend.iter_batches(val_texts):
        pooled = pool_activations(
            acts, attention_mask, granularity, cfg.extract.padding_side
        )
        for layer_pos, layer in enumerate(layers):
            if rows[layer] >= cfg.val_samples_per_layer:
                continue
            samples = pooled[:, layer_pos, :].to(dtype)
            collected[layer].append(samples)
            rows[layer] += samples.shape[0]
        if all(rows[layer] >= cfg.val_samples_per_layer for layer in layers):
            break
    for layer in layers:
        if rows[layer] == 0:
            continue
        chunk = torch.cat(collected[layer], dim=0)[: cfg.val_samples_per_layer]
        if not putter.broadcast(("val", chunk, layer)):
            return
    putter.broadcast(("val_done", None, 0))
    logger.info(
        "val broadcast done: %d prompts, %s samples/layer",
        len(val_texts),
        {layer: min(n, cfg.val_samples_per_layer) for layer, n in rows.items()},
    )


def start_thread_producer(
    cfg: StreamingConfig,
    model_name: str,
    out_queue: Any,
    stop_event: threading.Event,
    *,
    backend: ExtractionBackend | None = None,
    texts: list[str] | None = None,
    device: str | None = None,
    seed: int = 0,
) -> threading.Thread:
    """Run :func:`producer_loop` in a daemon thread (``num_producers=0`` mode)."""
    thread = threading.Thread(
        target=producer_loop,
        args=(cfg, model_name, 0, [out_queue], stop_event),
        kwargs={"backend": backend, "texts": texts, "device": device, "seed": seed},
        daemon=True,
        name="glp-producer",
    )
    thread.start()
    return thread


# ── consumer ─────────────────────────────────────────────────────────────────


def _chunk_to_samples(chunk: torch.Tensor, layer: int) -> list[dict[str, torch.Tensor]]:
    """Expand a ``(N, D)`` chunk into ActDataset-shaped sample dicts."""
    layer_idx = torch.tensor(layer, dtype=torch.long)
    return [
        {"activations": chunk[i].float()[None, :], "layer_idx": layer_idx}
        for i in range(chunk.shape[0])
    ]


class StreamingActDataset(IterableDataset[dict[str, torch.Tensor]]):
    """Rank-local stream of shuffled activation samples.

    Pulls :data:`ChunkMsg` items from a :class:`ChunkSource` and mixes them in a
    fixed-capacity in-memory buffer (a pre-allocated ``(capacity, D)`` tensor in
    the wire dtype, so 500k samples at D=2048 bf16 is ~2 GB). Chunks arrive
    layer-blocked; the buffer fills to ``min_fill`` before yielding and then
    evicts uniformly-random slots (a ``torch.Generator`` seeded ``seed`` makes
    the order deterministic), which decorrelates layers and prompts. Yielded
    samples have exactly the shape :class:`~glp.dataset.act_dataset.ActDataset`
    produces — ``{"activations": (1, D) float32, "layer_idx": long}`` — so
    :class:`~glp.dataset.act_dataset.ActivationCollator` is reused unchanged.

    ``__iter__`` returns one persistent generator: re-entering the epoch loop
    continues the stream rather than restarting it. Use ``num_workers=0`` — the
    queue must be consumed in-process.
    """

    def __init__(
        self,
        source: ChunkSource,
        num_producers: int,
        buffer_size: int,
        min_fill: float = 0.5,
        seed: int = 0,
        stall_timeout_s: float = 600.0,
        startup_timeout_s: float = 3600.0,
    ) -> None:
        self.source = source
        self.num_producers = max(num_producers, 1)
        self.buffer_size = buffer_size
        self.min_fill = min_fill
        self.seed = seed
        self.stall_timeout_s = stall_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self._gen: Iterator[dict[str, torch.Tensor]] | None = None
        self._pending: list[tuple[torch.Tensor, int]] = []  # chunks seen early
        self._val_taken = False
        self._ended = 0  # producers that sent "end" (cycle=False mode)
        self._started = False  # first chunk received -> producers are alive

    # ── message plumbing ──────────────────────────────────────────────────────

    def _get(self) -> ChunkMsg:
        """Blocking get on the startup budget until the first chunk, then the stall one."""
        timeout = self.stall_timeout_s if self._started else self.startup_timeout_s
        try:
            msg = self.source.get(timeout=timeout)
        except queue.Empty:
            phase = "producer startup" if not self._started else "training"
            raise RuntimeError(
                f"no activation chunk arrived within {timeout}s during {phase} — "
                "producers are stalled or dead"
            ) from None
        self._started = True
        return msg

    @staticmethod
    def _check_error(msg: ChunkMsg) -> None:
        if msg[0] == "error":
            raise RuntimeError(f"producer failed: {msg[2]}")

    def wait_for_val(self) -> list[dict[str, torch.Tensor]]:
        """Collect the validation broadcast (must be called before iterating).

        Blocks until producer 0's ``"val_done"`` arrives; train chunks that other
        producers interleave meanwhile are kept for the training stream. Returns
        the val set as a list of ActDataset-shaped sample dicts.
        """
        if self._val_taken:
            raise RuntimeError("wait_for_val() may only be called once")
        self._val_taken = True
        val_samples: list[dict[str, torch.Tensor]] = []
        while True:
            msg = self._get()
            self._check_error(msg)
            tag, chunk, meta = msg
            if tag == "val_done":
                logger.info("received val set: %d samples", len(val_samples))
                return val_samples
            if tag == "val":
                if chunk is None:
                    raise RuntimeError("malformed 'val' message: missing chunk")
                val_samples.extend(_chunk_to_samples(chunk, int(meta)))
            elif tag == "train":
                if chunk is None:
                    raise RuntimeError("malformed 'train' message: missing chunk")
                self._pending.append((chunk, int(meta)))
            elif tag == "end":
                raise RuntimeError(
                    "a producer ended before the val broadcast completed — "
                    "val_num_prompts is likely larger than the corpus"
                )
            else:
                raise RuntimeError(f"unexpected message tag {tag!r}")

    def _next_train_chunk(self) -> tuple[torch.Tensor, int] | None:
        """Blocking: next train chunk, or None once all producers have ended.

        Raises the stall error if nothing arrives within ``stall_timeout_s``.
        Used only for the initial buffer warm-up; steady state uses
        :meth:`_poll_train_chunk` so a transient empty queue drains the buffer
        instead of blocking.
        """
        if self._pending:
            return self._pending.pop(0)
        while True:
            msg = self._get()
            self._check_error(msg)
            tag, chunk, meta = msg
            if tag == "train":
                if chunk is None:
                    raise RuntimeError("malformed 'train' message: missing chunk")
                return chunk, int(meta)
            if tag == "end":
                self._ended += 1
                if self._ended >= self.num_producers:
                    return None
                continue
            raise RuntimeError(
                f"unexpected message tag {tag!r} in the training stream "
                "(was wait_for_val() called before iterating?)"
            )

    def _poll_train_chunk(
        self, timeout: float
    ) -> tuple[torch.Tensor, int] | object | None:
        """Non-fatal timed get: ``(chunk, layer)`` | :data:`_EMPTY` | ``None``.

        Returns ``_EMPTY`` if no chunk arrived within ``timeout`` (queue momentarily
        empty — the caller drains the buffer), or ``None`` once all producers have
        ended. Never raises on timeout (that is the buffer-drain fallback's job).
        """
        if self._pending:
            return self._pending.pop(0)
        try:
            msg = self.source.get(timeout=timeout)
        except queue.Empty:
            return _EMPTY
        self._check_error(msg)
        tag, chunk, meta = msg
        if tag == "train":
            if chunk is None:
                raise RuntimeError("malformed 'train' message: missing chunk")
            return chunk, int(meta)
        if tag == "end":
            self._ended += 1
            return None if self._ended >= self.num_producers else _EMPTY
        raise RuntimeError(
            f"unexpected message tag {tag!r} in the training stream "
            "(was wait_for_val() called before iterating?)"
        )

    # ── shuffle-buffer iteration ──────────────────────────────────────────────

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self._gen is None:
            self._gen = self._iterate()
        return self._gen

    def _iterate(self) -> Iterator[dict[str, torch.Tensor]]:
        gen = torch.Generator().manual_seed(self.seed)
        buf_acts: torch.Tensor | None = None  # (capacity, D), wire dtype
        buf_layers = torch.empty(self.buffer_size, dtype=torch.long)
        fill = 0
        min_fill_rows = max(1, int(self.buffer_size * self.min_fill))
        _poll = min(_DRAIN_POLL_S, self.stall_timeout_s)
        stream_ended = False

        def ingest(
            chunk: torch.Tensor, layer: int
        ) -> Iterator[dict[str, torch.Tensor]]:
            """Fold a chunk into the buffer, yielding evicted samples once warm."""
            nonlocal buf_acts, fill
            if buf_acts is None:
                buf_acts = torch.empty(
                    self.buffer_size, chunk.shape[1], dtype=chunk.dtype
                )
            offset = 0
            n = chunk.shape[0]
            # 1) fill empty capacity without yielding
            if fill < self.buffer_size:
                take = min(n, self.buffer_size - fill)
                buf_acts[fill : fill + take] = chunk[offset : offset + take]
                buf_layers[fill : fill + take] = layer
                fill += take
                offset += take
            # 2) replace random slots, yielding the evicted samples
            remaining = n - offset
            if remaining > 0 and fill >= min_fill_rows:
                slots = torch.randperm(fill, generator=gen)[:remaining]
                evicted_acts = buf_acts[slots].clone()
                evicted_layers = buf_layers[slots].clone()
                buf_acts[slots] = chunk[offset:]
                buf_layers[slots] = layer
                for i in range(evicted_acts.shape[0]):
                    yield {
                        "activations": evicted_acts[i].float()[None, :],
                        "layer_idx": evicted_layers[i].clone(),
                    }

        def drain_one() -> dict[str, torch.Tensor]:
            """Evict one random live sample WITHOUT inserting a new one.

            Used when the queue is momentarily empty: keeps the rank fed from the
            buffer (a producer hiccup becomes a buffer draw, not a blocked rank
            that desyncs DDP). Each sample is still emitted at most once — it is
            swap-removed from the live ``[0:fill]`` region.
            """
            nonlocal fill
            if buf_acts is None:  # unreachable: drain_one is only called with fill > 0
                raise RuntimeError("drain_one called before the buffer was allocated")
            j = int(torch.randint(fill, (1,), generator=gen).item())
            # clone before the swap below overwrites slot j: .float() is a no-op
            # view when the buffer is already float32, so without the clone the
            # yielded sample would alias the slot and be mutated by the swap.
            sample = {
                "activations": buf_acts[j].clone().float()[None, :],
                "layer_idx": buf_layers[j].clone(),
            }
            fill -= 1
            buf_acts[j] = buf_acts[fill]
            buf_layers[j] = buf_layers[fill]
            return sample

        # Warm-up: block-fill to min_fill (deterministic; this is also where a
        # genuine startup stall raises). ingest() yields nothing until the buffer
        # crosses min_fill.
        while fill < min_fill_rows and not stream_ended:
            item = self._next_train_chunk()
            if item is None:
                stream_ended = True
                break
            chunk, layer = item
            yield from ingest(chunk, layer)

        # Steady state: pull a chunk with a bounded poll. If the queue is empty
        # past the poll, drain the buffer to stay fed; only raise once the buffer
        # itself is exhausted (a genuinely dead/stalled producer).
        waited = 0.0
        while not stream_ended:
            item = self._poll_train_chunk(_poll)
            if item is None:  # all producers ended
                break
            if item is not _EMPTY:
                waited = 0.0
                chunk, layer = item  # type: ignore[misc]
                yield from ingest(chunk, layer)
            elif fill > 0:
                waited = 0.0
                yield drain_one()
            else:
                waited += _poll
                if waited >= self.stall_timeout_s:
                    raise RuntimeError(
                        f"no activation chunk arrived within {self.stall_timeout_s}s "
                        "and the shuffle buffer is empty — producers are stalled or dead"
                    )

        # Final drain: emit whatever remains (single-pass mode).
        while fill > 0:
            yield drain_one()
