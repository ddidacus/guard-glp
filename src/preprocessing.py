from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from typing import Any

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

logger = logging.getLogger(__name__)

NUM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


# ── filtering helpers ────────────────────────────────────────────────────────


def verify_moderation(entry: dict[str, Any]) -> bool:
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


def wildchat_clean_conversation(sample: dict[str, Any]) -> dict[str, Any]:
    conversation = [
        {"content": e["content"], "role": e["role"]} for e in sample["conversation"]
    ]
    sample["conversation"] = conversation
    return sample


def sample_sanitize_wildguard(sample: dict[str, Any]) -> bool:
    return (
        sample["adversarial"] is not True
        and sample["prompt_harm_label"] != "harmful"
        and sample["response_harm_label"] != "harmful"
    )


def sample_has_valid_conversation(sample: dict[str, Any]) -> bool:
    conv = sample.get("conversation")
    if not conv:
        return False
    return all(
        turn.get("role") is not None and turn.get("content") is not None
        for turn in conv
    )


# ── conversation formatting ──────────────────────────────────────────────────


def sample_format_conversation_wildjb(sample: dict[str, Any]) -> dict[str, Any]:
    txt_field = "vanilla" if sample["adversarial"] is None else "adversarial"
    sample["conversation"] = [
        {"content": sample[txt_field], "role": "user"},
        {"content": sample["completion"], "role": "assistant"},
    ]
    return sample


def sample_format_conversation_wildjb_eval(sample: dict[str, Any]) -> dict[str, Any]:
    # The "eval" config of allenai/wildjailbreak has no "vanilla"/"completion"
    # columns (only "adversarial" prompts), so it gets a single-turn conversation.
    sample["conversation"] = [
        {"content": sample["adversarial"], "role": "user"},
    ]
    return sample


def sample_format_conversation_wildguard(sample: dict[str, Any]) -> dict[str, Any]:
    sample["conversation"] = [
        {"content": sample["prompt"], "role": "user"},
        {"content": sample["response"], "role": "assistant"},
    ]
    return sample


# ── embedding ──────────────────────────────────────────────────────────────


def conversation_to_text(conversation: list[dict[str, Any]]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in conversation)


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer for embed + similarity."""

    def __init__(self, model_id: str, max_length: int = 512) -> None:
        logger.info("Loading embedding model %s …", model_id)
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

    def similarity(self, emb_a: Any, emb_b: Any) -> Any:
        return self._model.similarity(emb_a, emb_b)

    def unload(self) -> None:
        del self._model
        torch.cuda.empty_cache()


# ── tokenisation ─────────────────────────────────────────────────────────────


def sample_n_tokens(entry: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
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


def label_dataset_sample(sample: dict[str, Any], label: str) -> dict[str, Any]:
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
        sanitize_fn: Callable[[dict[str, Any]], bool] | None = None,
        format_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.label = label
        self._sanitize_fn = sanitize_fn
        self._format_fn = format_fn

    def sanitize(self) -> SourceHFDataset:
        if self._sanitize_fn is None:
            return self
        before = len(self.hf_dataset)
        self.hf_dataset = self.hf_dataset.filter(self._sanitize_fn, num_proc=NUM_CPUS)
        logger.info("%s: retained %.2f", self.label, len(self.hf_dataset) / before)
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
            logger.info(
                "%s: dropped %s rows with null fields", self.label, f"{dropped:,}"
            )
        return self

    def add_data_label(self) -> SourceHFDataset:
        self.hf_dataset = self.hf_dataset.map(
            lambda x, _l=self.label: label_dataset_sample(x, _l),
            num_proc=NUM_CPUS,
        )
        return self


# ── combined dataset ────────────────────────────────────────────────────────


class LSHKNN:
    """Random-hyperplane locality sensitive hashing for approximate near-duplicate lookup.

    Embeddings are bucketed by the sign pattern of their projection onto a
    fixed set of random hyperplanes; only embeddings that land in the same
    bucket are ever compared, turning an O(N^2) all-pairs similarity search
    into an approximate O(N) one.
    """

    def __init__(self, embedding_dim: int, num_buckets: int, device: str = "cpu"):
        self.device = device
        self.embedding_dim = embedding_dim
        # 2^H -> num_buckets possible buckets
        self.hidden_dim = max(
            1, int(torch.log2(torch.tensor(float(max(num_buckets, 2)))).item())
        )
        self.hyperplanes = torch.randn(
            embedding_dim, self.hidden_dim, device=self.device
        )  # E, H
        self._bit_weights = (2 ** torch.arange(self.hidden_dim, device=self.device)).to(
            torch.int64
        )
        self.reservoir: dict[int, torch.Tensor] = {}
        self.reservoir_indices: dict[int, list[int]] = {}

    def batch_hash(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        projections = X @ self.hyperplanes  # (B, E) @ (E, H) -> (B, H)
        binary_projections = (projections > 0).to(torch.int64)  # (B, H)
        return (binary_projections * self._bit_weights).sum(dim=1)  # (B,)

    def reservoir_push(self, X: torch.Tensor, indices: list[int] | None = None) -> None:
        X = X.to(self.device)
        X_keys = self.batch_hash(X)  # (B,)
        if indices is None:
            indices = list(range(len(X)))
        for key, x, idx in zip(X_keys.tolist(), X, indices):
            if key not in self.reservoir:
                self.reservoir[key] = x.unsqueeze(0)  # 1, E
                self.reservoir_indices[key] = [idx]
            else:
                self.reservoir[key] = torch.cat(
                    (self.reservoir[key], x.unsqueeze(0))
                )  # N, E
                self.reservoir_indices[key].append(idx)

    def batch_reservoir_get_knn(
        self, X: torch.Tensor
    ) -> list[tuple[int, torch.Tensor, list[int]] | None]:
        """Return, per row of X, the (bucket key, embeddings, indices) currently
        in its bucket, or None if the bucket is empty."""
        X = X.to(self.device)
        X_keys = self.batch_hash(X)  # (B,)
        return [
            (key, self.reservoir[key], self.reservoir_indices[key])
            if key in self.reservoir
            else None
            for key in X_keys.tolist()
        ]


class CombinedHFDataset:
    """Concatenated dataset with dedup, decontamination, and Hub push."""

    def __init__(self, datasets: list[Dataset]) -> None:
        self.hf_dataset = concatenate_datasets(datasets)
        logger.info("Total samples: %s", f"{len(self.hf_dataset):,}")

    def deduplicate_semantic(
        self,
        embedding_model: EmbeddingModel,
        embedding_batch_size: int = 1024,
        embedding_hidden_dim: int = 4096,
        sim_threshold: float = 0.99,
        samples_per_bucket: int = 512,
        device: str = "cpu",
    ) -> CombinedHFDataset:
        chunk_size = embedding_batch_size
        num_buckets = max(2, len(self.hf_dataset) // samples_per_bucket)
        self.lshknn = LSHKNN(
            embedding_dim=embedding_hidden_dim, num_buckets=num_buckets, device=device
        )

        # 1) embed everything and fill the LSH reservoir
        logger.info("Embedding %s samples …", f"{len(self.hf_dataset):,}")
        embeddings = torch.empty((len(self.hf_dataset), embedding_hidden_dim))

        for start in tqdm(
            range(0, len(self.hf_dataset), chunk_size),
            desc="  Embedding",
            leave=False,
        ):
            end = min(start + chunk_size, len(self.hf_dataset))
            chunk_convs = self.hf_dataset[start:end]["conversation"]
            chunk_texts = [conversation_to_text(c) for c in chunk_convs]
            chunk_emb = torch.as_tensor(
                embedding_model.embed(chunk_texts, batch_size=embedding_batch_size),
                dtype=torch.float32,
            )  # N, E
            embeddings[start:end] = chunk_emb
            self.lshknn.reservoir_push(chunk_emb, indices=list(range(start, end)))

        # 2) walk each bucket and drop near-duplicates, keeping the first
        # occurrence of each near-duplicate cluster.
        logger.info("Checking %s buckets for near-duplicates …", f"{len(self.lshknn.reservoir):,}")
        knn_results = self.lshknn.batch_reservoir_get_knn(embeddings)

        keep_mask = [True] * len(self.hf_dataset)
        seen_keys: set[int] = set()
        for result in knn_results:
            if result is None:
                continue
            key, bucket_embeddings, bucket_indices = result
            if key in seen_keys:
                continue
            seen_keys.add(key)

            kept_embeddings: list[torch.Tensor] = []
            for idx, emb in zip(bucket_indices, bucket_embeddings):
                emb = emb.unsqueeze(0)
                if kept_embeddings:
                    sims = embedding_model.similarity(emb, torch.stack(kept_embeddings))
                    if sims.max().item() > sim_threshold:
                        keep_mask[idx] = False
                        continue
                kept_embeddings.append(emb.squeeze(0))

        n_dupes = keep_mask.count(False)
        logger.info("Found %s near-duplicates", f"{n_dupes:,}")

        self.hf_dataset = self.hf_dataset.filter(
            lambda _, idx: keep_mask[idx], with_indices=True, num_proc=NUM_CPUS
        )
        logger.info("Samples after semantic dedup: %s", f"{len(self.hf_dataset):,}")
        return self

    def deduplicate_str_match(self) -> CombinedHFDataset:
        logger.info("Computing conversation hashes …")
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

        logger.info("Identifying duplicates …")
        seen: set[str] = set()
        keep_mask: list[bool] = []
        for h in ds["_hash"]:
            keep_mask.append(h not in seen)
            seen.add(h)

        n_dupes = keep_mask.count(False)
        logger.info("Found %s duplicates", f"{n_dupes:,}")

        ds = ds.filter(
            lambda _, idx: keep_mask[idx], with_indices=True, num_proc=NUM_CPUS
        )
        self.hf_dataset = ds.remove_columns(["_hash"])
        logger.info("Samples after dedup: %s", f"{len(self.hf_dataset):,}")
        return self

    def decontaminate(
        self,
        model: EmbeddingModel,
        reference: Dataset,
        threshold: float = 0.95,
        batch_size: int = 32,
        chunk_size: int = 8192,
    ) -> CombinedHFDataset:
        logger.info("Embedding %s reference samples …", f"{len(reference):,}")
        ref_texts = [conversation_to_text(c) for c in reference["conversation"]]
        ref_emb = model.embed(ref_texts, batch_size=batch_size, show_progress_bar=True)

        logger.info(
            "Embedding %s dataset samples & checking similarity …",
            f"{len(self.hf_dataset):,}",
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

        logger.info("Flagged %s contaminated samples", f"{len(contaminated):,}")
        model.unload()

        self.hf_dataset = self.hf_dataset.filter(
            lambda _, idx: idx not in contaminated,
            with_indices=True,
            num_proc=NUM_CPUS,
        )
        logger.info("Samples after decontamination: %s", f"{len(self.hf_dataset):,}")
        return self

    def save_to_disk(self, path: str) -> None:
        logger.info("Saving to %s …", path)
        self.hf_dataset.save_to_disk(path)
        logger.info("Done — saved to %s", path)

    def push_to_hf(self, repo_id: str, md_card: str, private: bool = False) -> None:
        logger.info("Pushing to %s …", repo_id)
        self.hf_dataset.push_to_hub(repo_id, private=private)
        HfApi().upload_file(
            path_or_fileobj=md_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        logger.info("Done — pushed to %s", repo_id)
