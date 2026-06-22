"""Prepare the Guard-GLP benign training dataset.

Loads LMSYS Chat 1M, WildChat, WildChat-4.8M, and WildGuardMix, filters out
harmful/adversarial samples, normalises conversation format, de-duplicates,
de-contaminates against WildJailbreak using Qwen3-Embedding-8B embeddings,
and pushes the result to the Hub.
"""

from __future__ import annotations

import os
from collections import Counter

import fire
from datasets import load_dataset

from src.preprocessing import (
    CombinedHFDataset,
    EmbeddingModel,
    SourceHFDataset,
    label_dataset_sample,
    remove_useless_columns,
    sample_format_conversation_wildguard,
    sample_format_conversation_wildjb,
    sample_has_valid_conversation,
    sample_sanitize_wildguard,
    verify_moderation,
    wildchat_clean_conversation,
)

NUM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-8B"


def main(
    repo_id: str = "ddidacus/guard-glp-benign",
    private: bool = False,
    output_dir: str | None = None,
    push_to_hub: bool = False,
    embed_batch_size: int = 256,
    embed_max_length: int = 512,
    sim_threshold: float = 0.95,
) -> None:
    # 1) Load & wrap raw datasets
    print("Loading datasets …")
    sources = [
        SourceHFDataset(
            load_dataset("lmsys/lmsys-chat-1m")["train"],
            label="lmsys",
            sanitize_fn=verify_moderation,
        ),
        SourceHFDataset(
            load_dataset("allenai/WildChat")["train"],
            label="wildchat",
            sanitize_fn=verify_moderation,
            format_fn=wildchat_clean_conversation,
        ),
        SourceHFDataset(
            load_dataset("allenai/WildChat-4.8M")["train"],
            label="wildchat_4m",
            sanitize_fn=verify_moderation,
            format_fn=wildchat_clean_conversation,
        ),
        SourceHFDataset(
            load_dataset("allenai/wildguardmix", "wildguardtrain")["train"],
            label="wildguard",
            sanitize_fn=sample_sanitize_wildguard,
            format_fn=sample_format_conversation_wildguard,
        ),
    ]

    # 2) Per-source pipeline: sanitize → format → drop nulls → label
    print("Processing sources …")
    processed = []
    for src in sources:
        src.sanitize().format_conversation().drop_nulls().add_data_label()
        processed.append(remove_useless_columns(src.hf_dataset))

    # 3) Combine, deduplicate, decontaminate
    combined = CombinedHFDataset(processed)
    combined.deduplicate()

    print("Preparing WildJailbreak reference …")
    wildjailbreak = load_dataset(
        "allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False
    )
    wildjb_ref = wildjailbreak["train"].map(sample_format_conversation_wildjb)
    wildjb_ref = wildjb_ref.filter(sample_has_valid_conversation, num_proc=NUM_CPUS)
    wildjb_ref = remove_useless_columns(
        wildjb_ref.map(
            lambda x: label_dataset_sample(x, "wildjailbreak"),
            num_proc=NUM_CPUS,
        )
    )
    print(f"WildJailbreak reference: {len(wildjb_ref):,} samples")

    embed_model = EmbeddingModel(EMBED_MODEL_ID, max_length=embed_max_length)
    combined.decontaminate(
        embed_model,
        wildjb_ref,
        threshold=sim_threshold,
        batch_size=embed_batch_size,
    )

    # 4) Save to disk and/or push to Hub
    if output_dir:
        combined.save_to_disk(output_dir)

    if push_to_hub:
        composition = Counter(combined.hf_dataset["origin"])
        comp_table = "\n".join(
            f"| {name} | {count:,} |" for name, count in composition.items()
        )

        card = f"""\
---
license: mit
---
# Guard-GLP Benign Conversations

Sanitized collection of benign multi-turn conversations, using **train splits only**.

## Composition
| Source | Samples |
|--------|---------|
{comp_table}
| **Total** | **{len(combined.hf_dataset):,}** |

## Sanitization
- **LMSYS Chat 1M, WildChat & WildChat-4.8M**: filtered via OpenAI moderation labels (all flagged categories removed)
- **WildGuardMix**: removed adversarial and harmful-labeled prompts/responses
- Exact de-duplication across all sources
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
