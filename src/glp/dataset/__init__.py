"""Dataset manager: build and load trainer-ready activation datasets.

Write side (the *manager*): :func:`build_shard` / :func:`finalize` turn a
``(dataset, model, layers, granularity)`` spec into per-layer dataset directories
with ``dtype.txt``, ``rep_statistics.pt`` and ``manifest.json``. Read side (the
*loader*, ported from the reference trainer): :func:`load_activation_dataset` /
:func:`get_activation_dataloader` / :class:`ActDataset`.
"""

from glp.dataset.act_dataset import (
    ActDataset,
    ActivationCollator,
    get_activation_dataloader,
    load_activation_dataset,
)
from glp.dataset.backends import (
    BatchActs,
    ExtractionBackend,
    HFBaukitBackend,
    VLLMNNSightBackend,
    make_backend,
)
from glp.dataset.builder import (
    BuildConfig,
    DatasetConfig,
    ExtractConfig,
    FilterConfig,
    build_shard,
    dataset_config_from_dict,
    extract_config_from_dict,
    finalize,
    resolve_layers,
    storage_dtype,
)
from glp.dataset.loader import load_texts
from glp.dataset.stats import RunningStats
from glp.dataset.streaming import (
    ChunkMsg,
    ChunkSource,
    QueueChunkSource,
    StreamingActDataset,
    StreamingConfig,
    producer_loop,
    start_thread_producer,
)

__all__ = [
    "ActDataset",
    "ActivationCollator",
    "BatchActs",
    "BuildConfig",
    "ChunkMsg",
    "ChunkSource",
    "DatasetConfig",
    "ExtractConfig",
    "ExtractionBackend",
    "FilterConfig",
    "HFBaukitBackend",
    "QueueChunkSource",
    "RunningStats",
    "StreamingActDataset",
    "StreamingConfig",
    "VLLMNNSightBackend",
    "build_shard",
    "dataset_config_from_dict",
    "extract_config_from_dict",
    "finalize",
    "get_activation_dataloader",
    "load_activation_dataset",
    "load_texts",
    "make_backend",
    "producer_loop",
    "resolve_layers",
    "start_thread_producer",
    "storage_dtype",
]
