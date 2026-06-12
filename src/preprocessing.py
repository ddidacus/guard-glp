from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable

import torch
from datasets import (
    Dataset,
    Features,
    Value,
    concatenate_datasets,
)
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

NUM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


# ── filtering helpers ────────────────────────────────────────────────────────


def verify_moderation(entry: dict) -> bool:
    if not entry.get("openai_moderation"):
        return False
    for entity in entry["openai_moderation"]:
        if entity is None:
            return False
        categories = entity.get("categories")
        if categories is None:
            return False
        for key in categories:
            if categories[key]:
                return False
    return True


def wildchat_clean_conversation(sample: dict) -> dict:
    conversation = [
        {"content": e["content"], "role": e["role"]}
        for e in sample["conversation"]
    ]
    sample["conversation"] = conversation
    return sample


def sample_sanitize_wildguard(sample: dict) -> bool:
    return (
        sample["adversarial"] is not True
        and sample["prompt_harm_label"] != "harmful"
        and sample["response_harm_label"] != "harmful"
    )


def sample_has_valid_conversation(sample: dict) -> bool:
    conv = sample.get("conversation")
    if not conv:
        return False
    return all(
        turn.get("role") is not None and turn.get("content") is not None
        for turn in conv
    )


# ── conversation formatting ──────────────────────────────────────────────────


def sample_format_conversation_wildjb(sample: dict) -> dict:
    txt_field = "vanilla" if sample["adversarial"] is None else "adversarial"
    sample["conversation"] = [
        {"content": sample[txt_field], "role": "user"},
        {"content": sample["completion"], "role": "assistant"},
    ]
    return sample


def sample_format_conversation_wildguard(sample: dict) -> dict:
    sample["conversation"] = [
        {"content": sample["prompt"], "role": "user"},
        {"content": sample["response"], "role": "assistant"},
    ]
    return sample


# ── embedding ──────────────────────────────────────────────────────────────


def conversation_to_text(conversation: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in conversation)


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer for embed + similarity."""

    def __init__(self, model_id: str, max_length: int = 512) -> None:
        print(f"  Loading embedding model {model_id} …")
        self._model = SentenceTransformer(
            model_id, model_kwargs={"torch_dtype": torch.float16}
        )
        self._model.max_seq_length = max_length

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ):
        return self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress_bar
        )

    def similarity(self, emb_a, emb_b):
        return self._model.similarity(emb_a, emb_b)

    def unload(self) -> None:
        del self._model
        torch.cuda.empty_cache()


# ── tokenisation ─────────────────────────────────────────────────────────────


def sample_n_tokens(entry: dict, tokenizer: object) -> dict:
    tokenized = tokenizer.apply_chat_template(
        entry["conversation"],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    entry["n_tokens"] = tokenized["input_ids"].shape[-1]
    return entry


def get_n_tokens(dataset: Dataset, tokenizer: object) -> int:
    data_tokens = dataset.map(
        lambda x: sample_n_tokens(x, tokenizer), num_proc=NUM_CPUS
    )
    return int(torch.tensor(data_tokens["n_tokens"]).sum().item())


# ── labelling / cleanup ─────────────────────────────────────────────────────


def label_dataset_sample(sample: dict, label: str) -> dict:
    sample["origin"] = label
    return sample


def remove_useless_columns(dataset: Dataset) -> Dataset:
    columns_to_drop = [
        c for c in dataset.column_names if c not in ("origin", "conversation")
    ]
    return dataset.remove_columns(columns_to_drop)


# ── source dataset ──────────────────────────────────────────────────────────


class SourceHFDataset:
    """Wraps an HF Dataset with per-source sanitization and formatting."""

    def __init__(
        self,
        hf_dataset: Dataset,
        label: str,
        sanitize_fn: Callable[[dict], bool] | None = None,
        format_fn: Callable[[dict], dict] | None = None,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.label = label
        self._sanitize_fn = sanitize_fn
        self._format_fn = format_fn

    def sanitize(self) -> SourceHFDataset:
        if self._sanitize_fn is None:
            return self
        before = len(self.hf_dataset)
        self.hf_dataset = self.hf_dataset.filter(
            self._sanitize_fn, num_proc=NUM_CPUS
        )
        print(
            f"  {self.label}: retained "
            f"{len(self.hf_dataset) / before:.2f}"
        )
        return self

    def format_conversation(self) -> SourceHFDataset:
        if self._format_fn is None:
            return self
        target_features = self.hf_dataset.features.copy()
        target_features["conversation"] = [
            {"content": Value("string"), "role": Value("string")}
        ]
        self.hf_dataset = self.hf_dataset.map(
            self._format_fn,
            num_proc=NUM_CPUS,
            features=Features(target_features),
        )
        return self

    def drop_nulls(self) -> SourceHFDataset:
        before = len(self.hf_dataset)
        self.hf_dataset = self.hf_dataset.filter(
            sample_has_valid_conversation, num_proc=NUM_CPUS
        )
        dropped = before - len(self.hf_dataset)
        if dropped:
            print(f"  {self.label}: dropped {dropped:,} rows with null fields")
        return self

    def add_data_label(self) -> SourceHFDataset:
        self.hf_dataset = self.hf_dataset.map(
            lambda x, _l=self.label: label_dataset_sample(x, _l),
            num_proc=NUM_CPUS,
        )
        return self


# ── combined dataset ────────────────────────────────────────────────────────


class CombinedHFDataset:
    """Concatenated dataset with dedup, decontamination, and Hub push."""

    def __init__(self, datasets: list[Dataset]) -> None:
        self.hf_dataset = concatenate_datasets(datasets)
        print(f"Total samples: {len(self.hf_dataset):,}")

    def deduplicate(self) -> CombinedHFDataset:
        print("  Computing conversation hashes …")
        ds = self.hf_dataset.map(
            lambda x: {
                "_hash": hashlib.sha256(
                    json.dumps(
                        x["conversation"], sort_keys=True, ensure_ascii=False
                    ).encode()
                ).hexdigest()
            },
            num_proc=NUM_CPUS,
        )

        print("  Identifying duplicates …")
        seen: set[str] = set()
        keep_mask: list[bool] = []
        for h in ds["_hash"]:
            keep_mask.append(h not in seen)
            seen.add(h)

        n_dupes = keep_mask.count(False)
        print(f"  Found {n_dupes:,} duplicates")

        ds = ds.filter(
            lambda _, idx: keep_mask[idx], with_indices=True, num_proc=NUM_CPUS
        )
        self.hf_dataset = ds.remove_columns(["_hash"])
        print(f"  Samples after dedup: {len(self.hf_dataset):,}")
        return self

    def decontaminate(
        self,
        model: EmbeddingModel,
        reference: Dataset,
        threshold: float = 0.95,
        batch_size: int = 32,
        chunk_size: int = 8192,
    ) -> CombinedHFDataset:
        print(f"  Embedding {len(reference):,} reference samples …")
        ref_texts = [
            conversation_to_text(c) for c in reference["conversation"]
        ]
        ref_emb = model.embed(
            ref_texts, batch_size=batch_size, show_progress_bar=True
        )

        print(
            f"  Embedding {len(self.hf_dataset):,} dataset samples "
            f"& checking similarity …"
        )
        contaminated: set[int] = set()

        for start in tqdm(
            range(0, len(self.hf_dataset), chunk_size),
            desc="  Decontaminating",
            leave=False,
        ):
            end = min(start + chunk_size, len(self.hf_dataset))
            chunk_convs = self.hf_dataset[start:end]["conversation"]
            chunk_texts = [conversation_to_text(c) for c in chunk_convs]
            chunk_emb = model.embed(chunk_texts, batch_size=batch_size)
            sims = model.similarity(chunk_emb, ref_emb)
            max_sims = sims.max(dim=1).values
            for j in range(len(max_sims)):
                if max_sims[j].item() > threshold:
                    contaminated.add(start + j)

        print(f"  Flagged {len(contaminated):,} contaminated samples")
        model.unload()

        self.hf_dataset = self.hf_dataset.filter(
            lambda _, idx: idx not in contaminated,
            with_indices=True,
            num_proc=NUM_CPUS,
        )
        print(f"  Samples after decontamination: {len(self.hf_dataset):,}")
        return self

    def save_to_disk(self, path: str) -> None:
        print(f"Saving to {path} …")
        self.hf_dataset.save_to_disk(path)
        print(f"Done — saved to {path}")

    def push_to_hf(
        self, repo_id: str, md_card: str, private: bool = False
    ) -> None:
        print(f"Pushing to {repo_id} …")
        self.hf_dataset.push_to_hub(repo_id, private=private)
        HfApi().upload_file(
            path_or_fileobj=md_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Done — pushed to {repo_id}")