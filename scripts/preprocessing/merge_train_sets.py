"""Prepare the Guard-GLP benign training dataset (per-shard process + finalize).

Loads LMSYS Chat 1M, WildChat, WildChat-4.8M, and WildGuardMix, filters out
harmful/adversarial samples, normalises conversation format, optionally
de-duplicates, de-contaminates against WildJailbreak using Qwen3-Embedding-8B
embeddings, and pushes the result to the Hub.

Two subcommands, following the repo's ``fire`` two-pass convention so the work
can be split across GPUs:

    # pass 1: one process per shard, each pinned to its own GPU, independently
    # filtering/formatting/decontaminating its slice of the corpus
    python scripts/preprocessing/merge_train_sets.py shard \\
        --shard_id=0 --num_shards=4 --shard_dir=data/guardglp_benign_shards

    # pass 2: merge shards (single process, no GPU needed unless --deduplicate)
    python scripts/preprocessing/merge_train_sets.py finalize \\
        --shard_dir=data/guardglp_benign_shards --num_shards=4 \\
        --output_dir=data/guardglp_benign --push_to_hub

On a cluster node these are driven for you by
``scripts/preprocessing/merge_train_sets.sh`` (backgrounds ``NUM_THREADS``
shard workers, one per GPU index, then runs ``finalize`` once all exit 0).
"""

# 1. convert all samples to qwen embeddings
# 2. locality sensitive hashing
# 3. approximate KNN for de-duplication

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import fire
import torch
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

from glp.preprocessing import (
    CombinedHFDataset,
    EmbeddingModel,
    SourceHFDataset,
    get_n_tokens,
    label_dataset_sample,
    remove_useless_columns,
    sample_format_conversation_wildguard,
    sample_format_conversation_wildjb,
    sample_format_conversation_wildjb_eval,
    sample_has_valid_conversation,
    sample_sanitize_wildguard,
    verify_moderation,
    wildchat_clean_conversation,
)

NUM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
TOKENIZER_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def _shard_dir(shard_dir: str, shard_id: int) -> Path:
    return Path(shard_dir) / f"shard_{shard_id}"


def _ref_emb_path(shard_dir: str) -> Path:
    """Cached WildJailbreak reference embedding, shared by all shard workers."""
    return Path(shard_dir) / "wildjb_ref_emb.pt"


def _ref_emb_part_path(shard_dir: str, ref_shard_id: int) -> Path:
    """One GPU's slice of the reference embedding (merged into `_ref_emb_path`)."""
    return Path(shard_dir) / f"wildjb_ref_emb.part_{ref_shard_id}.pt"


def _load_sources(shard_id: int, num_shards: int) -> list[Dataset]:
    print("Loading datasets …")
    sources = [
        SourceHFDataset(
            load_dataset("lmsys/lmsys-chat-1m")["train"],
            label="lmsys",
            sanitize_fn=verify_moderation,
        ),
        SourceHFDataset(
            load_dataset("allenai/WildChat-4.8M")["train"],
            label="wildchat_4m",
            sanitize_fn=verify_moderation,
            format_fn=wildchat_clean_conversation,
        ),
    ]

    # slice each source into this shard's contiguous chunk before doing any
    # (expensive) filtering/formatting work on it

    if num_shards > 1:
        for src in sources:
            src.hf_dataset = src.hf_dataset.shard(
                num_shards=num_shards, index=shard_id, contiguous=True
            )

    print(f"[shard {shard_id}/{num_shards}] Processing sources …")
    processed = []
    for src in sources:
        src.sanitize().format_conversation().drop_nulls().add_data_label()
        processed.append(remove_useless_columns(src.hf_dataset))
    return processed


def _build_wildjb_reference() -> Dataset:
    print("[+] Preparing WildJailbreak reference …")

    def _format_and_filter_wildjb(ds, format_fn):
        ds = ds.map(format_fn)
        ds = ds.filter(sample_has_valid_conversation, num_proc=NUM_CPUS)
        return ds.select_columns(["conversation"])

    wildjailbreak_train = load_dataset(
        "allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False
    )["train"]
    wildjailbreak_eval = load_dataset(
        "allenai/wildjailbreak", "eval", delimiter="\t", keep_default_na=False
    )["train"]

    wildjb_ref = concatenate_datasets(
        [
            _format_and_filter_wildjb(
                wildjailbreak_train, sample_format_conversation_wildjb
            ),
            _format_and_filter_wildjb(
                wildjailbreak_eval, sample_format_conversation_wildjb_eval
            ),
        ]
    )
    wildjb_ref = remove_useless_columns(
        wildjb_ref.map(
            lambda x: label_dataset_sample(x, "wildjailbreak"),
            num_proc=NUM_CPUS,
        )
    )
    print(f"WildJailbreak reference: {len(wildjb_ref):,} samples")
    return wildjb_ref


def embed_reference(
    shard_dir: str,
    ref_shard_id: int = 0,
    ref_num_shards: int = 1,
    embed_batch_size: int = 256,
    embed_max_length: int = 512,
) -> None:
    """Embed the WildJailbreak reference and cache it for all shard workers.

    The reference is identical across shards, so embedding it in every shard
    worker wastes ~8x the GPU-hours. Run this before the shard array; each
    ``shard`` call then loads the cached tensor instead of recomputing it.

    Embedding the reference is itself embarrassingly parallel, so it can be
    split across GPUs: launch ``ref_num_shards`` workers (one per GPU), each
    with a distinct ``ref_shard_id``. Each writes a contiguous slice to a part
    file; ``merge_reference`` concatenates the parts in order into the final
    cache. With ``ref_num_shards=1`` (default) it writes the final cache
    directly, preserving the single-GPU behaviour.
    """
    Path(shard_dir).mkdir(parents=True, exist_ok=True)

    wildjb_ref = _build_wildjb_reference()
    if ref_num_shards > 1:
        wildjb_ref = wildjb_ref.shard(
            num_shards=ref_num_shards, index=ref_shard_id, contiguous=True
        )

    tag = f"embed_reference {ref_shard_id}/{ref_num_shards}"
    embed_model = EmbeddingModel(EMBED_MODEL_ID, max_length=embed_max_length)
    print(f"[{tag}] Embedding {len(wildjb_ref):,} reference samples …")
    ref_emb = embed_model.embed_conversations(
        wildjb_ref["conversation"],
        batch_size=embed_batch_size,
        show_progress_bar=True,
    )
    embed_model.unload()

    out_path = (
        _ref_emb_path(shard_dir)
        if ref_num_shards == 1
        else _ref_emb_part_path(shard_dir, ref_shard_id)
    )
    torch.save(ref_emb, out_path)
    print(f"[{tag}] Cached reference embedding {tuple(ref_emb.shape)} → {out_path}")


def merge_reference(shard_dir: str, ref_num_shards: int) -> None:
    """Concatenate the per-GPU reference embedding parts into the final cache.

    Parts must be joined in ``ref_shard_id`` order so row *i* of the merged
    tensor still corresponds to reference sample *i* (``shard(contiguous=True)``
    slices the reference in order).
    """
    parts = []
    for i in range(ref_num_shards):
        part_path = _ref_emb_part_path(shard_dir, i)
        if not part_path.exists():
            raise FileNotFoundError(
                f"missing reference part {part_path}; did embed_reference "
                f"shard {i}/{ref_num_shards} finish?"
            )
        parts.append(torch.load(part_path))

    ref_emb = torch.cat(parts, dim=0)
    ref_path = _ref_emb_path(shard_dir)
    torch.save(ref_emb, ref_path)
    print(
        f"[merge_reference] Merged {ref_num_shards} parts → {tuple(ref_emb.shape)} "
        f"at {ref_path}"
    )
    for i in range(ref_num_shards):
        _ref_emb_part_path(shard_dir, i).unlink()


def shard(
    shard_id: int,
    num_shards: int,
    shard_dir: str,
    embed_batch_size: int = 256,
    embed_max_length: int = 512,
    sim_threshold: float = 0.99,
) -> None:
    """Process and decontaminate one shard of the corpus (pin one GPU per call).

    Independent per shard: filtering, conversation formatting, and
    decontamination against WildJailbreak. Semantic de-duplication needs a
    global view of the corpus, so it is not run here — pass ``--deduplicate``
    to ``finalize`` instead.

    Reuses the cached reference embedding from ``embed_reference`` when present
    (``<shard_dir>/wildjb_ref_emb.pt``), falling back to embedding the reference
    in-process if the cache is missing.
    """

    processed = _load_sources(shard_id, num_shards)
    combined = CombinedHFDataset(processed)

    ref_path = _ref_emb_path(shard_dir)
    ref_emb = None
    wildjb_ref = None
    if ref_path.exists():
        print(f"[shard {shard_id}] Loading cached reference embedding {ref_path} …")
        ref_emb = torch.load(ref_path)
    else:
        print(
            f"[shard {shard_id}] No cached reference at {ref_path}; embedding in-process"
        )
        wildjb_ref = _build_wildjb_reference()

    print(f"[shard {shard_id}] Decontamination …")
    embed_model = EmbeddingModel(EMBED_MODEL_ID, max_length=embed_max_length)
    combined.decontaminate(
        embed_model,
        reference=wildjb_ref,
        ref_emb=ref_emb,
        threshold=sim_threshold,
        batch_size=embed_batch_size,
    )

    out_dir = _shard_dir(shard_dir, shard_id)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    combined.save_to_disk(str(out_dir))
    print(
        f"[shard {shard_id}] Done — {len(combined.hf_dataset):,} samples saved to {out_dir}"
    )


def finalize(
    shard_dir: str,
    num_shards: int,
    repo_id: str = "ddidacus/guard-glp-benign",
    private: bool = False,
    output_dir: str | None = None,
    push_to_hub: bool = False,
    deduplicate: bool = False,
    embed_batch_size: int = 256,
    embed_max_length: int = 512,
    sim_threshold: float = 0.99,
    semantic_dedup_samples_per_bucket: int = 512,
    tokenizer_id: str = TOKENIZER_MODEL_ID,
    test_size: float = 0.01,
) -> None:
    """Merge shards produced by ``shard``, optionally de-duplicate, and push."""

    print(f"[+] Loading {num_shards} shard(s) from {shard_dir} …")
    shards = [load_from_disk(str(_shard_dir(shard_dir, i))) for i in range(num_shards)]
    combined = CombinedHFDataset(shards)

    # near-duplicate removal via LSH-bucketed approximate KNN over embeddings;
    # needs the full merged corpus, so it only runs here, never per-shard

    if deduplicate:
        embed_model = EmbeddingModel(EMBED_MODEL_ID, max_length=embed_max_length)
        combined.deduplicate_semantic(
            embed_model,
            embedding_batch_size=embed_batch_size,
            sim_threshold=sim_threshold,
            samples_per_bucket=semantic_dedup_samples_per_bucket,
        )

    # count tokens of the final dataset

    print(f"[+] Counting tokens with {tokenizer_id} …")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    n_tokens = get_n_tokens(combined.hf_dataset, tokenizer)
    print(f"Total tokens: {n_tokens:,}")

    # held-out test split

    composition = Counter(combined.hf_dataset["origin"])
    print(f"[+] Holding out {test_size:.0%} as a uniformly sampled test split …")
    combined.hf_dataset = combined.hf_dataset.train_test_split(
        test_size=test_size, seed=42
    )
    print(
        f"Train: {len(combined.hf_dataset['train']):,}  "
        f"Test: {len(combined.hf_dataset['test']):,}"
    )

    # save and push to the hub

    if output_dir:
        combined.save_to_disk(output_dir)

    if push_to_hub:
        comp_table = "\n".join(
            f"| {name} | {count:,} |" for name, count in composition.items()
        )

        semantic_dedup_step = (
            "5. Semantic near-duplicate removal (LSH-bucketed approximate KNN over embeddings)\n"
            if deduplicate
            else ""
        )
        card = f"""\
---
license: mit
---
# Guard-GLP Benign Conversations

Sanitized collection of benign multi-turn conversations, using **train splits only**.

## Pipeline
1. Load LMSYS Chat 1M and WildChat-4.8M (sharded across {num_shards} worker(s))
2. Filter harmful/adversarial samples via OpenAI moderation labels
3. Normalise conversation format across sources
4. Exact de-duplication (conversation hash)
{semantic_dedup_step}6. De-contaminate against WildJailbreak (train + eval configs)
7. Hold out a uniformly sampled test split

## Composition
| Source | Samples |
|--------|---------|
{comp_table}
| **Total** | **{sum(composition.values()):,}** |

## Splits
| Split | Samples |
|-------|---------|
| train | {len(combined.hf_dataset["train"]):,} |
| test | {len(combined.hf_dataset["test"]):,} |

Test split is a uniformly sampled {test_size:.0%} holdout (seed 42).

Total tokens ({tokenizer_id} chat template): **{n_tokens:,}**

## Sanitization
- **LMSYS Chat 1M, WildChat & WildChat-4.8M**: filtered via OpenAI moderation labels (all flagged categories removed)
- **WildGuardMix**: removed adversarial and harmful-labeled prompts/responses
- Exact de-duplication across all sources
{f"- Semantic near-duplicate removal via LSH-bucketed approximate KNN (Qwen3-Embedding-8B, cosine similarity > {sim_threshold} removed)" if deduplicate else ""}
- Embedding-based de-contamination against WildJailbreak (Qwen3-Embedding-8B, cosine similarity > {sim_threshold} removed)

Only benign conversations are retained.
"""

        combined.push_to_hf(repo_id, md_card=card, private=private)

    if not output_dir and not push_to_hub:
        print(
            "Warning: neither --output_dir nor --push_to_hub specified; "
            "dataset was processed but not saved."
        )


if __name__ == "__main__":
    fire.Fire(
        {
            "embed_reference": embed_reference,
            "merge_reference": merge_reference,
            "shard": shard,
            "finalize": finalize,
        }
    )
