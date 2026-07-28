"""Shared, reusable cache for extracted LLM activations used by the detection evals.

The three detection scripts (reconstruction-error, diff-of-means, linear probe) all
score the SAME LLM activations — activations depend only on the dataset split, the
LLM, the hooked layers, and the token pooling, NOT on the GLP or the detection method.
Previously each run cached under its own ``out_dir`` and so re-extracted the identical
activations for every config (full / useronly / off-the-shelf / probe / diffmean).

This module holds one shared cache, keyed by exactly the fields that determine the
activations, so a split is extracted once and reused by every later run. The key is
human-readable (``dataset__llm__L14__mean__test__0.th``) so the cache is easy to
inspect and clean by hand.
"""

import logging
import re
from collections.abc import Callable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Shared location on the (shared) filesystem, alongside the eval outputs.
DEFAULT_CACHE_DIR = Path("results/_activation_cache")


def _slug(value: str) -> str:
    """Filesystem-safe token: keep alnum/._-, collapse everything else to '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def cache_key(
    dataset: str,
    llm_model_id: str,
    layers: list[int],
    token_pooling: str,
    split: str,
    shard: int,
) -> str:
    """Human-readable filename encoding everything the activations depend on."""
    layers_tag = "L" + "-".join(str(int(x)) for x in layers)
    parts = [
        _slug(dataset),
        _slug(llm_model_id),
        layers_tag,
        _slug(token_pooling),
        _slug(split),
        f"s{int(shard)}",
    ]
    return "__".join(parts) + ".th"


def cached_activations(
    *,
    dataset: str,
    llm_model_id: str,
    layers: list[int],
    token_pooling: str,
    split: str,
    shard: int,
    extract: Callable[[], torch.Tensor],
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> torch.Tensor:
    """Return cached (N, L, D) CPU activations for this split, extracting if absent.

    ``extract`` is called (and its result cached) only on a miss. The key deliberately
    includes ``llm_model_id``, ``layers`` and ``token_pooling`` so runs that differ in
    any of those never collide (e.g. the off-the-shelf baseline on base Llama vs. the
    guard-glp-benign runs on Instruct).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / cache_key(
        dataset, llm_model_id, layers, token_pooling, split, shard
    )
    if path.exists():
        logger.info("act-cache hit: %s", path.name)
        return torch.load(path, map_location="cpu", weights_only=True)
    logger.info("act-cache miss: %s — extracting", path.name)
    acts = extract().cpu()
    # atomic-ish write so a crash mid-save can't leave a truncated cache entry
    tmp = path.with_suffix(".th.tmp")
    torch.save(acts, tmp)
    tmp.replace(path)
    return acts
