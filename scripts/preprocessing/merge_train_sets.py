"""Prepare the Guard-GLP benign training dataset.

Loads LMSYS Chat 1M, WildChat, WildChat-4.8M, and WildGuardMix, filters out
harmful/adversarial samples, normalises conversation format, de-duplicates,
de-contaminates against WildJailbreak using Qwen3-Embedding-8B embeddings,
and pushes the result to the Hub.
"""

# 1. convert all samples to qwen embeddings
# 2. locality sensitive hashing
# 3. approximate KNN for de-duplication

from __future__ import annotations

import os
from collections import Counter

import fire
from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from src.preprocessing import (
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


def main(
    repo_id: str = "ddidacus/guard-glp-benign",
    private: bool = False,
    output_dir: str | None = None,
    push_to_hub: bool = False,
    embed_batch_size: int = 256,
    embed_max_length: int = 512,
    sim_threshold: float = 0.99,
    semantic_dedup_samples_per_bucket: int = 512,
    tokenizer_id: str = TOKENIZER_MODEL_ID,
    test_size: float = 0.01,
) -> None:
    
    # sources
    
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

    # format and drop invalid

    print("[+] Processing sources …")
    processed = []
    for src in sources:
        src.sanitize().format_conversation().drop_nulls().add_data_label()
        processed.append(remove_useless_columns(src.hf_dataset))

    # merge and deduplicate by string matching

    combined = CombinedHFDataset(processed)

    # embedding model, reused for semantic dedup and decontamination below

    embed_model = EmbeddingModel(EMBED_MODEL_ID, max_length=embed_max_length)

    # near-duplicate removal via LSH-bucketed approximate KNN over embeddings

    combined.deduplicate_semantic(
        embed_model,
        embedding_batch_size=embed_batch_size,
        sim_threshold=sim_threshold,
        samples_per_bucket=semantic_dedup_samples_per_bucket,
    )

    # unify wildjailbreak reference to decontaminate from

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
            _format_and_filter_wildjb(wildjailbreak_train, sample_format_conversation_wildjb),
            _format_and_filter_wildjb(wildjailbreak_eval, sample_format_conversation_wildjb_eval),
        ]
    )
    wildjb_ref = remove_useless_columns(
        wildjb_ref.map(
            lambda x: label_dataset_sample(x, "wildjailbreak"),
            num_proc=NUM_CPUS,
        )
    )
    print(f"WildJailbreak reference: {len(wildjb_ref):,} samples")

    # clean up train data from wildjailbreak

    print("[+] Decontamination ")
    combined.decontaminate(
        embed_model,
        wildjb_ref,
        threshold=sim_threshold,
        batch_size=embed_batch_size,
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

        card = f"""\
---
license: mit
---
# Guard-GLP Benign Conversations

Sanitized collection of benign multi-turn conversations, using **train splits only**.

## Pipeline
1. Load LMSYS Chat 1M and WildChat-4.8M
2. Filter harmful/adversarial samples via OpenAI moderation labels
3. Normalise conversation format across sources
4. Exact de-duplication (conversation hash)
5. Semantic near-duplicate removal (LSH-bucketed approximate KNN over embeddings)
6. De-contaminate against WildJailbreak (train + eval configs)
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
- Semantic near-duplicate removal via LSH-bucketed approximate KNN (Qwen3-Embedding-8B, cosine similarity > {sim_threshold} removed)
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
    fire.Fire(main)
