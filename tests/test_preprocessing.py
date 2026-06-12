"""Tests for src.preprocessing – SourceHFDataset & CombinedHFDataset.

All fixtures use synthetic in-memory datasets — no network, no GPU required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from datasets import Dataset

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

N = 20


# ── synthetic dataset builders ──────────────────────────────────────────────


def _make_lmsys_like() -> Dataset:
    """LMSYS-shaped: conversation + openai_moderation.

    Rows 0-3:  flagged moderation  → filtered by verify_moderation
    Rows 4-5:  null assistant content → caught by drop_nulls
    Rows 6-19: clean
    """
    rows = []
    for i in range(N):
        flagged = i < 4
        null_content = i in (4, 5)
        rows.append(
            {
                "conversation": [
                    {"role": "user", "content": f"question {i}"},
                    {
                        "role": "assistant",
                        "content": None if null_content else f"answer {i}",
                    },
                ],
                "openai_moderation": [
                    {"categories": {"violence": flagged}},
                ],
            }
        )
    return Dataset.from_list(rows)


def _make_wildchat_like() -> Dataset:
    """WildChat-shaped: conversation with extra keys + openai_moderation.

    Same flagged / null layout as LMSYS; conversation turns carry extra
    keys (turn_identifier, model) that wildchat_clean_conversation strips.
    """
    rows = []
    for i in range(N):
        flagged = i < 4
        null_content = i in (4, 5)
        rows.append(
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": f"hi {i}",
                        "turn_identifier": i,
                        "model": "gpt-4",
                    },
                    {
                        "role": "assistant",
                        "content": None if null_content else f"hello {i}",
                        "turn_identifier": i + 1,
                        "model": "gpt-4",
                    },
                ],
                "openai_moderation": [
                    {"categories": {"violence": flagged}},
                ],
            }
        )
    return Dataset.from_list(rows)


def _make_wildguard_like() -> Dataset:
    """WildGuard-shaped: prompt, response, adversarial, harm labels.

    Rows 0-2:  adversarial=True       → filtered by sanitize
    Rows 3-4:  prompt_harm="harmful"   → filtered by sanitize
    Row  5:    response_harm="harmful" → filtered by sanitize
    Rows 6-7:  response=None           → caught by drop_nulls (after format)
    Rows 8-19: clean
    """
    rows = []
    for i in range(N):
        rows.append(
            {
                "prompt": f"prompt {i}",
                "response": None if i in (6, 7) else f"response {i}",
                "adversarial": i < 3,
                "prompt_harm_label": "harmful" if i in (3, 4) else "unharmful",
                "response_harm_label": "harmful" if i == 5 else "unharmful",
            }
        )
    return Dataset.from_list(rows)


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def lmsys() -> Dataset:
    return _make_lmsys_like()


@pytest.fixture()
def wildchat() -> Dataset:
    return _make_wildchat_like()


@pytest.fixture()
def wildguard() -> Dataset:
    return _make_wildguard_like()


@pytest.fixture()
def processed_datasets(
    lmsys: Dataset, wildchat: Dataset, wildguard: Dataset
) -> list[Dataset]:
    sources = [
        SourceHFDataset(lmsys, "lmsys", sanitize_fn=verify_moderation),
        SourceHFDataset(
            wildchat,
            "wildchat",
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
        assert len(src.hf_dataset) == N - 4
        for row in src.hf_dataset:
            assert verify_moderation(row)

    def test_wildguard_filters(self, wildguard: Dataset) -> None:
        src = SourceHFDataset(
            wildguard, "wildguard", sanitize_fn=sample_sanitize_wildguard
        )
        src.sanitize()
        assert len(src.hf_dataset) == N - 6
        for row in src.hf_dataset:
            assert sample_sanitize_wildguard(row)

    def test_no_fn_is_noop(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys")
        src.sanitize()
        assert len(src.hf_dataset) == N

    def test_returns_self(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys", sanitize_fn=verify_moderation)
        assert src.sanitize() is src


# ── SourceHFDataset.format_conversation ─────────────────────────────────────


class TestFormatConversation:
    def test_wildchat_strips_extra_keys(self, wildchat: Dataset) -> None:
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
    def test_removes_null_content(self, lmsys: Dataset) -> None:
        src = SourceHFDataset(lmsys, "lmsys", sanitize_fn=verify_moderation)
        src.sanitize()
        before = len(src.hf_dataset)
        src.drop_nulls()
        assert len(src.hf_dataset) == before - 2

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


# ── mock embedding helpers ──────────────────────────────────────────────────


def _make_mock_embedding_model(dim: int = 64) -> MagicMock:
    """Mock EmbeddingModel with random deterministic embeddings."""
    model = MagicMock(spec=EmbeddingModel)
    call_count = [0]

    def _embed(texts: list[str], **_kwargs: object) -> np.ndarray:
        call_count[0] += 1
        rng = np.random.default_rng(seed=42 + call_count[0])
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


# ── CombinedHFDataset.decontaminate ────────────────────────────────────────


class TestDecontaminate:
    def test_self_reference_removes_all(
        self, processed_datasets: list[Dataset]
    ) -> None:
        combined = CombinedHFDataset(processed_datasets)
        combined.deduplicate()
        before = len(combined.hf_dataset)

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

    def test_unrelated_reference_keeps_all(
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

    def test_calls_unload(self, processed_datasets: list[Dataset]) -> None:
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
        combined.decontaminate(
            model, dummy_ref, threshold=0.99, chunk_size=256
        )
        model.unload.assert_called_once()

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


# ── CombinedHFDataset.save_to_disk ────────────────────────────────────────


class TestSaveToDisk:
    def test_saves(
        self, processed_datasets: list[Dataset], tmp_path: object
    ) -> None:
        combined = CombinedHFDataset(processed_datasets)
        out = str(tmp_path)
        combined.save_to_disk(out)
        reloaded = Dataset.load_from_disk(out)
        assert len(reloaded) == len(combined.hf_dataset)
        assert set(reloaded.column_names) == set(
            combined.hf_dataset.column_names
        )


# ── CombinedHFDataset.push_to_hf ──────────────────────────────────────────


class TestPushToHf:
    def test_calls_hub_apis(
        self, processed_datasets: list[Dataset]
    ) -> None:
        combined = CombinedHFDataset(processed_datasets)
        with (
            patch.object(
                combined.hf_dataset, "push_to_hub"
            ) as mock_push,
            patch("src.preprocessing.HfApi") as mock_api_cls,
        ):
            combined.push_to_hf(
                "test/repo", md_card="# Card", private=True
            )
            mock_push.assert_called_once_with("test/repo", private=True)
            mock_api_cls.return_value.upload_file.assert_called_once()
            call_kwargs = mock_api_cls.return_value.upload_file.call_args
            assert call_kwargs.kwargs["path_in_repo"] == "README.md"
            assert call_kwargs.kwargs["repo_id"] == "test/repo"
