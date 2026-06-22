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
    finalize,
    storage_dtype,
)
from glp.dataset.loader import load_texts
from glp.dataset.stats import RunningStats

__all__ = [
    "ActDataset",
    "ActivationCollator",
    "BatchActs",
    "BuildConfig",
    "DatasetConfig",
    "ExtractConfig",
    "ExtractionBackend",
    "FilterConfig",
    "HFBaukitBackend",
    "RunningStats",
    "VLLMNNSightBackend",
    "build_shard",
    "finalize",
    "get_activation_dataloader",
    "load_activation_dataset",
    "load_texts",
    "make_backend",
    "storage_dtype",
]
