"""Config-driven text ingestion for activation extraction.

Loads an HF ``datasets`` split and turns each row into a single prompt string,
either by reading a plain ``text_field`` or by applying the tokenizer's chat
template over a ``conversation_field`` (WildChat / LMSYS style). Generic,
optional filters (column equality, char-length bounds), dedup and a global
``max_samples`` cap are applied before sharding the result across GPUs.

Dataset-specific cleaning recipes (e.g. WildChat toxicity filtering) are kept
out of here by design; they are added later as config + the richer helpers in
:mod:`preprocessing` (``SourceHFDataset`` / ``CombinedHFDataset``).
"""

import hashlib
import logging
from typing import TYPE_CHECKING, Any

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from glp.dataset.builder import DatasetConfig

logger = logging.getLogger(__name__)


def load_texts(
    cfg: "DatasetConfig",
    tokenizer: PreTrainedTokenizerBase,
    gpu_id: int,
    num_gpus: int,
) -> list[str]:
    """Load, format, filter and shard the configured dataset into texts.

    The returned list is this shard's slice (``texts[gpu_id::num_gpus]``) of the
    globally-capped corpus, so the per-GPU shards partition the same data.
    """
    dataset: Any = load_dataset(
        cfg.path, name=cfg.name, split=cfg.split, revision=cfg.revision
    )

    for filt in cfg.filters:
        dataset = dataset.filter(
            lambda row, col=filt.column, val=filt.equals: row[col] == val
        )

    if cfg.format == "chat":
        texts: list[Any] = [
            tokenizer.apply_chat_template(
                row[cfg.conversation_field],
                tokenize=False,
                add_generation_prompt=False,
            )
            for row in dataset
        ]
    elif cfg.format == "text":
        if cfg.text_field is None:
            raise ValueError("text_field is required when format='text'")
        texts = [row[cfg.text_field] for row in dataset]
    else:
        raise ValueError(f"unknown dataset format: {cfg.format!r}")

    cleaned: list[str] = [t for t in texts if isinstance(t, str) and t]
    if cfg.min_chars is not None:
        cleaned = [t for t in cleaned if len(t) >= cfg.min_chars]
    if cfg.max_chars is not None:
        cleaned = [t for t in cleaned if len(t) <= cfg.max_chars]

    if cfg.dedup:
        seen: set[str] = set()
        deduped: list[str] = []
        for text in cleaned:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest not in seen:
                seen.add(digest)
                deduped.append(text)
        logger.info("dedup: %d -> %d texts", len(cleaned), len(deduped))
        cleaned = deduped

    if cfg.max_samples is not None:
        cleaned = cleaned[: cfg.max_samples]

    return cleaned[gpu_id::num_gpus]
