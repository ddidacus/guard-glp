"""Build trainer-ready activation datasets from (dataset, model, layers).

Two passes, mirroring the repo's ``fire`` + ``.sh`` GPU-sharding convention:

* :func:`build_shard` (pass 1, one process per GPU): stream texts through an
  :class:`~glp.dataset.backends.ExtractionBackend`, pool each batch to the
  requested granularity, fold values into per-layer :class:`RunningStats`, and
  write each ``(D,)`` sample via a :class:`~glp.utils_acts.MemmapWriter` into
  ``shard_<g>/<gran>/layer_<NN>/``. A bounded queue between the producing
  backend and the consuming writer provides backpressure.
* :func:`finalize` (pass 2, single process): merge the shard memmaps into
  ``<base>/<gran>/layer_<NN>/`` (rename data files, re-offset indices), combine
  the partial stats into ``rep_statistics.pt`` via ``Normalizer.save_config``,
  and write ``dtype.txt`` + ``manifest.json``.

The GLP training loop is out of scope; the produced directories are consumed by
the in-repo loader in :mod:`glp.dataset.act_dataset`.
"""

import logging
import queue
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import OmegaConf

from glp.dataset.backends import BatchActs, ExtractionBackend, make_backend
from glp.dataset.loader import load_texts
from glp.dataset.manifest import build_manifest, write_manifest
from glp.dataset.stats import RunningStats
from glp.utils_acts import MemmapWriter, pool_activations

logger = logging.getLogger(__name__)


# ── configuration ────────────────────────────────────────────────────────────


@dataclass
class FilterConfig:
    column: str
    equals: Any


@dataclass
class DatasetConfig:
    path: str
    split: str = "train"
    name: str | None = None
    revision: str | None = None
    format: str = "text"  # "text" | "chat"
    conversation_field: str = "conversation"
    text_field: str | None = None
    # Chat-only: which part of the conversation to feed the model.
    #   "full" -> the whole conversation, chat-templated (add_generation_prompt=False).
    #   "user" -> only the first user turn as a standalone single-turn prompt,
    #             chat-templated with add_generation_prompt=True (the deployment
    #             screening position). See load_texts.
    prompt_view: str = "full"  # "full" | "user"
    filters: list[FilterConfig] = field(default_factory=list)
    min_chars: int | None = None
    max_chars: int | None = None
    dedup: bool = False
    max_samples: int | None = None


@dataclass
class ExtractConfig:
    layers: list[int]
    layer_prefix: str = "model.layers"
    retain: str = "output"  # "output" | "input"
    granularity: list[str] = field(default_factory=lambda: ["last"])
    max_length: int = 2048
    batch_size: int = 32
    dtype: str = "float32"  # "float32" | "bfloat16"
    padding_side: str = "right"
    # Whether the tokenizer adds special tokens (e.g. BOS). None -> resolved from the
    # dataset format in make_backend: chat-templated text already carries the
    # template's specials, so add_special_tokens=False for "chat" and True for "text".
    add_special_tokens: bool | None = None
    queue_maxsize: int = 16
    file_size: int = 33554432  # elements per memmap chunk
    tensor_parallel_size: int = 1  # vllm_nnsight only: GPUs per (single) shard


@dataclass
class BuildConfig:
    model_name: str
    output_dir: str
    dataset: DatasetConfig
    extract: ExtractConfig
    save_root: str = "."
    backend: str = "hf_baukit"
    # Number of data-parallel shards: = number of GPUs for `hf_baukit` (one shard
    # per GPU); = 1 for `vllm_nnsight`, which parallelizes a single shard across
    # GPUs via `extract.tensor_parallel_size` instead.
    num_gpus: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BuildConfig":
        raw = OmegaConf.load(Path(path))
        container = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(container, dict):
            raise ValueError(f"config root must be a mapping, got {type(container)}")
        return cls.from_dict({str(k): v for k, v in container.items()})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BuildConfig":
        ds = dict(data.get("dataset", {}))
        ex = dict(data.get("extract", {}))
        filters = [
            FilterConfig(column=f["column"], equals=f.get("equals"))
            for f in (ds.get("filters") or [])
        ]
        dataset = DatasetConfig(
            path=ds["path"],
            split=ds.get("split", "train"),
            name=ds.get("name"),
            revision=ds.get("revision"),
            format=ds.get("format", "text"),
            conversation_field=ds.get("conversation_field", "conversation"),
            text_field=ds.get("text_field"),
            prompt_view=ds.get("prompt_view", "full"),
            filters=filters,
            min_chars=ds.get("min_chars"),
            max_chars=ds.get("max_chars"),
            dedup=bool(ds.get("dedup", False)),
            max_samples=ds.get("max_samples"),
        )
        extract = ExtractConfig(
            layers=[int(layer) for layer in ex["layers"]],
            layer_prefix=ex.get("layer_prefix", "model.layers"),
            retain=ex.get("retain", "output"),
            granularity=list(ex.get("granularity", ["last"])),
            max_length=int(ex.get("max_length", 2048)),
            batch_size=int(ex.get("batch_size", 32)),
            dtype=ex.get("dtype", "float32"),
            padding_side=ex.get("padding_side", "right"),
            add_special_tokens=ex.get("add_special_tokens"),
            queue_maxsize=int(ex.get("queue_maxsize", 16)),
            file_size=int(ex.get("file_size", 33554432)),
            tensor_parallel_size=int(ex.get("tensor_parallel_size", 1)),
        )
        return cls(
            model_name=data["model_name"],
            output_dir=data["output_dir"],
            dataset=dataset,
            extract=extract,
            save_root=data.get("save_root", "."),
            backend=data.get("backend", "hf_baukit"),
            num_gpus=int(data.get("num_gpus", 1)),
        )


# ── dtype encoding ───────────────────────────────────────────────────────────


def storage_dtype(config_dtype: str) -> tuple[np.dtype[Any], str]:
    """Map a config dtype to its (numpy storage dtype, ``dtype.txt`` string).

    ``bfloat16`` has no numpy equivalent, so it is stored as ``int16`` (the same
    bytes) and reinterpreted by ``ActDataset`` via ``.view(torch.bfloat16)``.
    """
    if config_dtype == "float32":
        return np.dtype(np.float32), "float32"
    if config_dtype == "bfloat16":
        return np.dtype(np.int16), "int16"
    raise ValueError(f"unsupported dtype: {config_dtype!r} (use float32 | bfloat16)")


def _encode_rows(samples: torch.Tensor, config_dtype: str) -> npt.NDArray[Any]:
    """Encode ``(N, D)`` float samples to the on-disk numpy array."""
    if config_dtype == "bfloat16":
        return samples.to(torch.bfloat16).view(torch.int16).cpu().numpy()
    return samples.float().cpu().numpy()


# ── paths ────────────────────────────────────────────────────────────────────


def _shard_root(cfg: BuildConfig, gpu_id: int) -> Path:
    return Path(cfg.output_dir) / f"shard_{gpu_id}"


def _layer_dir(base: Path, granularity: str, layer: int) -> Path:
    return base / granularity / f"layer_{layer:02d}"


_DATA_FILE_RE = re.compile(r"data_\d+\.npy")


# ── pass 1: per-shard extraction ─────────────────────────────────────────────


def build_shard(
    cfg: BuildConfig,
    gpu_id: int,
    *,
    backend: ExtractionBackend | None = None,
    texts: list[str] | None = None,
    device: str | None = None,
) -> None:
    """Extract one data shard of activations.

    ``gpu_id`` is the shard index (strides the corpus via ``load_texts``), not a
    device id. ``device`` overrides the CUDA device selection (see
    :func:`~glp.dataset.backends.make_backend`); under a SLURM array task the one
    allocated GPU is pinned, so pass ``"cuda:0"``. ``backend`` and ``texts`` are
    injectable for testing; in production both are derived from ``cfg``.
    """
    if backend is None:
        backend, tokenizer = make_backend(cfg, gpu_id, device=device)
        if texts is None:
            texts = load_texts(cfg.dataset, tokenizer, gpu_id, cfg.num_gpus)
    if texts is None:
        raise ValueError("texts must be provided when backend is injected")

    shard_dir = _shard_root(cfg, gpu_id)
    logger.info("shard %d: extracting %d texts into %s", gpu_id, len(texts), shard_dir)
    _run_pipeline(cfg, backend, texts, shard_dir)
    logger.info("shard %d: done", gpu_id)


def _run_pipeline(
    cfg: BuildConfig, backend: ExtractionBackend, texts: list[str], shard_dir: Path
) -> None:
    np_dtype, _ = storage_dtype(cfg.extract.dtype)
    granularities = list(cfg.extract.granularity)
    layers = list(cfg.extract.layers)
    writers: dict[tuple[str, int], MemmapWriter] = {}
    stats: dict[tuple[str, int], RunningStats] = {}

    batch_queue: queue.Queue[BatchActs | None] = queue.Queue(
        maxsize=cfg.extract.queue_maxsize
    )
    errors: list[BaseException] = []

    def produce() -> None:
        try:
            for batch in backend.iter_batches(texts):
                batch_queue.put(batch)
        except BaseException as exc:  # noqa: BLE001 - re-raised on consumer side
            errors.append(exc)
        finally:
            batch_queue.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()

    while True:
        item = batch_queue.get()
        if item is None:
            break
        acts, attention_mask = item
        _consume_batch(
            cfg,
            acts,
            attention_mask,
            granularities,
            layers,
            shard_dir,
            np_dtype,
            writers,
            stats,
        )
    producer.join()
    if errors:
        raise errors[0]

    for writer in writers.values():
        writer.flush()
    for (granularity, layer), running in stats.items():
        running.save_partial(
            _layer_dir(shard_dir, granularity, layer) / "stats_partial.pt"
        )


def _consume_batch(
    cfg: BuildConfig,
    acts: torch.Tensor,
    attention_mask: torch.Tensor,
    granularities: list[str],
    layers: list[int],
    shard_dir: Path,
    np_dtype: np.dtype[Any],
    writers: dict[tuple[str, int], MemmapWriter],
    stats: dict[tuple[str, int], RunningStats],
) -> None:
    for granularity in granularities:
        # pooled: (N, L, D) for last/mean, (T, L, D) for all
        token_idx = cast(Literal["last", "mean", "all"], granularity)
        pooled = pool_activations(
            acts, attention_mask, token_idx, cfg.extract.padding_side
        )
        for layer_idx, layer in enumerate(layers):
            samples = pooled[:, layer_idx, :]  # (N, D)
            key = (granularity, layer)
            if key not in writers:
                layer_dir = _layer_dir(shard_dir, granularity, layer)
                writers[key] = MemmapWriter(
                    output_dir=layer_dir,
                    file_size=cfg.extract.file_size,
                    dtype=np_dtype,
                )
                stats[key] = RunningStats.zeros(int(samples.shape[1]))
            stats[key].update(samples.float())
            encoded = _encode_rows(samples, cfg.extract.dtype)
            for row in encoded:
                writers[key].write(np.ascontiguousarray(row))


# ── pass 2: merge shards ─────────────────────────────────────────────────────


def finalize(cfg: BuildConfig) -> None:
    """Merge all shards into trainer-ready per-(granularity, layer) datasets."""
    _, dtype_str = storage_dtype(cfg.extract.dtype)
    out = Path(cfg.output_dir)

    for granularity in cfg.extract.granularity:
        for layer in cfg.extract.layers:
            shard_dirs = sorted(out.glob(f"shard_*/{granularity}/layer_{layer:02d}"))
            if not shard_dirs:
                logger.warning(
                    "no shards found for granularity=%s layer=%d", granularity, layer
                )
                continue
            final_dir = _layer_dir(out, granularity, layer)
            if final_dir.exists():
                shutil.rmtree(final_dir)
            final_dir.mkdir(parents=True)

            num_samples = _merge_data(shard_dirs, final_dir)
            merged_stats = _merge_stats(shard_dirs)

            (final_dir / "dtype.txt").write_text(dtype_str + "\n")
            merged_stats.to_normalizer().save_config(final_dir)
            manifest = build_manifest(
                cfg, granularity, layer, num_samples, merged_stats.dim, dtype_str
            )
            write_manifest(final_dir, manifest)
            logger.info(
                "finalized %s: %d samples, dim %d, dtype %s",
                final_dir,
                num_samples,
                merged_stats.dim,
                dtype_str,
            )

    for shard in out.glob("shard_*"):
        if shard.is_dir():
            shutil.rmtree(shard)


def _merge_data(shard_dirs: list[Path], final_dir: Path) -> int:
    """Move each shard's data files into ``final_dir`` and re-offset indices."""
    global_file_count = 0
    merged_indices: list[npt.NDArray[np.uint64]] = []
    for shard in shard_dirs:
        indices_path = shard / "data_indices.npy"
        if not indices_path.exists():
            continue
        indices = np.load(indices_path).astype(np.uint64)
        data_files = sorted(
            p for p in shard.glob("data_*.npy") if _DATA_FILE_RE.fullmatch(p.name)
        )
        offset = global_file_count
        for data_file in data_files:
            shutil.move(
                str(data_file), str(final_dir / f"data_{global_file_count:04d}.npy")
            )
            global_file_count += 1
        if len(indices) > 0:
            indices = indices.copy()
            indices[:, 0] = indices[:, 0] + offset
            merged_indices.append(indices)

    if merged_indices:
        out_indices = np.concatenate(merged_indices, axis=0)
    else:
        out_indices = np.zeros((0, 3), dtype=np.uint64)
    np.save(final_dir / "data_indices.npy", out_indices.astype(np.uint64))
    return int(len(out_indices))


def _merge_stats(shard_dirs: list[Path]) -> RunningStats:
    merged: RunningStats | None = None
    for shard in shard_dirs:
        stats_path = shard / "stats_partial.pt"
        if not stats_path.exists():
            continue
        partial = RunningStats.load(stats_path)
        if merged is None:
            merged = partial
        else:
            merged.merge(partial)
    if merged is None:
        raise ValueError("no stats_partial.pt found in any shard")
    return merged
