"""Dataset manager: build and load trainer-ready activation datasets.

Write side (the *manager*): :func:`build_shard` / :func:`finalize` turn a
``(dataset, model, layers, granularity)`` spec into per-layer dataset directories
with ``dtype.txt``, ``rep_statistics.pt`` and ``manifest.json``. Read side (the
*loader*, ported from the reference trainer): :func:`load_activation_dataset` /
:func:`get_activation_dataloader` / :class:`ActDataset`.
"""

from glp.dataset.act_cache import cached_activations
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
from glp.dataset.eval_prompts import EvalPrompts, load_eval_prompts
from glp.dataset.loader import load_texts
from glp.dataset.ood_prompts import PromptPool, load_id_pool, load_ood_pool
from glp.dataset.stats import RunningStats

__all__ = [
    "ActDataset",
    "ActivationCollator",
    "BatchActs",
    "BuildConfig",
    "DatasetConfig",
    "EvalPrompts",
    "ExtractConfig",
    "ExtractionBackend",
    "FilterConfig",
    "HFBaukitBackend",
    "PromptPool",
    "RunningStats",
    "VLLMNNSightBackend",
    "build_shard",
    "cached_activations",
    "finalize",
    "get_activation_dataloader",
    "load_activation_dataset",
    "load_eval_prompts",
    "load_id_pool",
    "load_ood_pool",
    "load_texts",
    "make_backend",
    "storage_dtype",
]
