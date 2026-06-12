"""Tests for src.preprocessing – SourceHFDataset & CombinedHFDataset.

Loads 100-sample subsets of real HF Hub datasets.  Requires network access
and is slower than pure-unit tests; run with ``pytest -m slow`` to include
or ``pytest -m 'not slow'`` to skip.
"""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset, load_dataset

from src.preprocessing import (
    CombinedHFDataset,
    EmbeddingModel,
    SourceHFDataset,
    remove_useless_columns,
    sample_format_conversation_wildguard,
    sample_has_valid_conversation,
    sample_sanitize_wildguard,
    verify_moderation,
    wildchat_clean_conversation,
)

SUBSET = 20

pytestmark = pytest.mark.slow


def _stream_subset(name: str, subset: int, **kwargs: object) -> Dataset:
    stream = load_dataset(name, split="train", streaming=True, **kwargs)
    return Dataset.from_list(list(itertools.islice(stream, subset)))


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def lmsys() -> Dataset:
    return _stream_subset("lmsys/lmsys-chat-1m", SUBSET)


@pytest.fixture(scope="module")
def wildchat() -> Dataset:
    return _stream_subset("allenai/WildChat", SUBSET)


@pytest.fixture(scope="module")
def wildchat_4m() -> Dataset:
    return _stream_subset("allenai/WildChat-4.8M", SUBSET)


@pytest.fixture(scope="module")
def wildguard() -> Dataset:
    return _stream_subset(
        "allenai/wildguardmix", SUBSET, name="wildguardtrain"
    )


@pytest.fixture(scope="module")
def processed_datasets(
    lmsys: Dataset, wildchat: Dataset, wildchat_4m: Dataset, wildguard: Dataset
) -> list[Dataset]:
    sources = [
        SourceHFDataset(
            lmsys, "lmsys", sanitize_fn=verify_moderation
        ),
        SourceHFDataset(
            wildchat,
            "wildchat",
            sanitize_fn=verify_moderation,
            format_fn=wildchat_clean_conversation,
        ),
        SourceHFDataset(
            wildchat_4m,
            "wildchat_4m",
            sanitize_fn=verify_moderation,
            format_fn=wildchat_clean_conversation,
        ),
        SourceHFDataset(
            wildguard,
            "wildguard",
            sanitize_fn=sample_sanitize_wildguard,
            format_fn=sample_format_conversation_wildguard,
        ),
    ]
    processed: list[Dataset] = []
    for src in sources:
        src.sanitize().format_conversation().drop_nulls().add_data_label()
        processed.append(remove_useless_columns(src.hf_dataset))
    return processed


# ── SourceHFDataset.sanitize ────────────────────────────────────────────────


class TestSanitize:
    def test_moderation_filters(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys", sanitize_fn=verify_moderation)
        src.sanitize()
        assert len(src.hf_dataset) <= SUBSET
        for row in src.hf_dataset:
            assert verify_moderation(row)

    def test_wildguard_filters(self, wildguard: Dataset) -> None:
        src = SourceHFDataset(
            wildguard, "wildguard", sanitize_fn=sample_sanitize_wildguard
        )
        src.sanitize()
        assert len(src.hf_dataset) <= SUBSET
        for row in src.hf_dataset:
            assert sample_sanitize_wildguard(row)

    def test_no_fn_is_noop(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys")
        src.sanitize()
        assert len(src.hf_dataset) == SUBSET

    def test_returns_self(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys", sanitize_fn=verify_moderation)
        assert src.sanitize() is src


# ── SourceHFDataset.format_conversation ─────────────────────────────────────


class TestFormatConversation:
    def test_wildchat_normalises_turns(self, wildchat: Dataset) -> None:
        src = SourceHFDataset(
            wildchat,
            "wildchat",
            sanitize_fn=verify_moderation,
            format_fn=wildchat_clean_conversation,
        )
        src.sanitize().format_conversation()
        for row in src.hf_dataset:
            for turn in row["conversation"]:
                assert set(turn.keys()) == {"content", "role"}

    def test_wildchat_4m_normalises_turns(self, wildchat_4m: Dataset) -> None:
        src = SourceHFDataset(
            wildchat_4m,
            "wildchat_4m",
            sanitize_fn=verify_moderation,
            format_fn=wildchat_clean_conversation,
        )
        src.sanitize().format_conversation()
        for row in src.hf_dataset:
            for turn in row["conversation"]:
                assert set(turn.keys()) == {"content", "role"}

    def test_wildguard_builds_conversation(self, wildguard: Dataset) -> None:
        src = SourceHFDataset(
            wildguard,
            "wildguard",
            sanitize_fn=sample_sanitize_wildguard,
            format_fn=sample_format_conversation_wildguard,
        )
        src.sanitize().format_conversation()
        for row in src.hf_dataset:
            conv = row["conversation"]
            assert len(conv) == 2
            assert conv[0]["role"] == "user"
            assert conv[1]["role"] == "assistant"

    def test_no_fn_is_noop(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys")
        cols_before = set(lmsys.column_names)
        src.format_conversation()
        assert set(src.hf_dataset.column_names) == cols_before

    def test_returns_self(self, wildchat: Dataset) -> None:
        src = SourceHFDataset(
            wildchat, "wildchat", format_fn=wildchat_clean_conversation
        )
        assert src.format_conversation() is src


# ── SourceHFDataset.drop_nulls ──────────────────────────────────────────────


class TestDropNulls:
    def test_all_remaining_valid(self, wildguard: Dataset) -> None:
        src = SourceHFDataset(
            wildguard,
            "wildguard",
            sanitize_fn=sample_sanitize_wildguard,
            format_fn=sample_format_conversation_wildguard,
        )
        src.sanitize().format_conversation().drop_nulls()
        for row in src.hf_dataset:
            assert sample_has_valid_conversation(row)

    def test_size_does_not_grow(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys", sanitize_fn=verify_moderation)
        src.sanitize()
        before = len(src.hf_dataset)
        src.drop_nulls()
        assert len(src.hf_dataset) <= before

    def test_returns_self(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys")
        assert src.drop_nulls() is src


# ── SourceHFDataset.add_data_label ──────────────────────────────────────────


class TestAddDataLabel:
    def test_adds_origin_column(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys")
        src.add_data_label()
        assert "origin" in src.hf_dataset.column_names
        assert all(o == "lmsys" for o in src.hf_dataset["origin"])

    def test_label_matches_constructor(self, wildguard: Dataset) -> None:
        src = SourceHFDataset(wildguard, "wildguard")
        src.add_data_label()
        assert set(src.hf_dataset["origin"]) == {"wildguard"}

    def test_returns_self(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys")
        assert src.add_data_label() is src


# ── SourceHFDataset full-chain ──────────────────────────────────────────────


class TestSourceChaining:
    def test_full_pipeline(self, wildchat: Dataset) -> None:
        src = SourceHFDataset(
            wildchat,
            "wildchat",
            sanitize_fn=verify_moderation,
            format_fn=wildchat_clean_conversation,
        )
        result = (
            src.sanitize()
            .format_conversation()
            .drop_nulls()
            .add_data_label()
        )
        assert result is src
        assert "origin" in src.hf_dataset.column_names
        assert len(src.hf_dataset) > 0


# ── CombinedHFDataset.__init__ ──────────────────────────────────────────────


class TestCombinedInit:
    def test_concatenates(self, processed_datasets: list[Dataset]) -> None:
        combined = CombinedHFDataset(processed_datasets)
        expected = sum(len(ds) for ds in processed_datasets)
        assert len(combined.hf_dataset) == expected

    def test_schema(self, processed_datasets: list[Dataset]) -> None:
        combined = CombinedHFDataset(processed_datasets)
        assert set(combined.hf_dataset.column_names) == {
            "origin",
            "conversation",
        }


# ── CombinedHFDataset.deduplicate ──────────────────────────────────────────


class TestDeduplicate:
    def test_removes_exact_duplicates(
        self, processed_datasets: list[Dataset]
    ) -> None:
        doubled = processed_datasets + processed_datasets
        combined = CombinedHFDataset(doubled)
        before = len(combined.hf_dataset)
        combined.deduplicate()
        expected = sum(len(ds) for ds in processed_datasets)
        assert len(combined.hf_dataset) == expected
        assert len(combined.hf_dataset) < before

    def test_no_hash_column_leaks(
        self, processed_datasets: list[Dataset]
    ) -> None:
        combined = CombinedHFDataset(processed_datasets)
        combined.deduplicate()
        assert "_hash" not in combined.hf_dataset.column_names

    def test_returns_self(self, processed_datasets: list[Dataset]) -> None:
        combined = CombinedHFDataset(processed_datasets)
        assert combined.deduplicate() is combined


# ── CombinedHFDataset.decontaminate ────────────────────────────────────────


def _make_mock_embedding_model(dim: int = 64) -> MagicMock:
    """Build a mock EmbeddingModel that produces deterministic embeddings."""
    model = MagicMock(spec=EmbeddingModel)

    def _embed(texts: list[str], **_kwargs: object) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        return rng.standard_normal((len(texts), dim)).astype(np.float32)

    def _similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> torch.Tensor:
        a = torch.from_numpy(emb_a)
        b = torch.from_numpy(emb_b)
        a = a / a.norm(dim=1, keepdim=True)
        b = b / b.norm(dim=1, keepdim=True)
        return a @ b.T

    model.embed.side_effect = _embed
    model.similarity.side_effect = _similarity
    model.unload.return_value = None
    return model


class TestDecontaminate:
    def test_self_reference_removes_samples(
        self, processed_datasets: list[Dataset]
    ) -> None:
        combined = CombinedHFDataset(processed_datasets)
        combined.deduplicate()
        before = len(combined.hf_dataset)

        # Mock that returns identical embeddings for both ref and dataset,
        # so every sample has cosine-similarity 1.0 with the reference.
        model = MagicMock(spec=EmbeddingModel)
        model.embed.return_value = np.ones(
            (before, 8), dtype=np.float32
        )
        model.similarity.side_effect = (
            lambda a, b: torch.from_numpy(a) @ torch.from_numpy(b).T
        )
        model.unload.return_value = None

        combined.decontaminate(
            model, combined.hf_dataset, threshold=0.5, chunk_size=256
        )
        assert len(combined.hf_dataset) == 0

    def test_unrelated_reference_keeps_samples(
        self, processed_datasets: list[Dataset]
    ) -> None:
        combined = CombinedHFDataset(processed_datasets)
        combined.deduplicate()
        before = len(combined.hf_dataset)
        dummy_ref = Dataset.from_dict(
            {
                "conversation": [
                    [
                        {"role": "user", "content": "xyzzy foobar"},
                        {"role": "assistant", "content": "plugh quux"},
                    ]
                ],
                "origin": ["dummy"],
            }
        )
        model = _make_mock_embedding_model()
        combined.decontaminate(
            model, dummy_ref, threshold=0.99, chunk_size=256
        )
        assert len(combined.hf_dataset) == before

    def test_returns_self(self, processed_datasets: list[Dataset]) -> None:
        combined = CombinedHFDataset(processed_datasets)
        dummy_ref = Dataset.from_dict(
            {
                "conversation": [
                    [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ]
                ],
                "origin": ["dummy"],
            }
        )
        model = _make_mock_embedding_model()
        assert (
            combined.decontaminate(
                model, dummy_ref, threshold=0.99, chunk_size=256
            )
            is combined
        )