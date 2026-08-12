"""CPU-only tests for the text loader's ``prompt_view`` handling (no network).

``load_texts`` is exercised with a fake tokenizer and a monkeypatched
``load_dataset`` so no model or dataset download is needed. Covers the
``full`` vs ``user`` chat views and the defensive skip of non-user opening turns.
"""

from typing import Any, cast

import pytest
from transformers import PreTrainedTokenizerBase

from glp.dataset import loader as loader_mod
from glp.dataset.builder import DatasetConfig, FilterConfig


class _FakeTokenizer:
    """Records apply_chat_template calls and renders a deterministic string."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        rendered = "".join(f"<{m['role']}>{m['content']}" for m in conversation)
        if add_generation_prompt:
            rendered += "<gen>"
        return rendered


def FakeTokenizer() -> PreTrainedTokenizerBase:  # noqa: N802 - factory reads as a class
    """A duck-typed stand-in for the only tokenizer method ``load_texts`` uses."""
    return cast(PreTrainedTokenizerBase, _FakeTokenizer())


CONV = [
    {"role": "user", "content": "U0"},
    {"role": "assistant", "content": "A0"},
    {"role": "user", "content": "U1"},
    {"role": "assistant", "content": "A1"},
]


class FakeHFDataset(list):  # type: ignore[type-arg]
    """A list that mimics the slice of ``datasets.Dataset`` ``load_texts`` uses.

    ``filter`` only accepts the batched, single-column call the loader makes, so a
    regression back to a row-wise predicate (which decodes every column) fails here.
    """

    def filter(
        self,
        function: Any,
        input_columns: str | None = None,
        batched: bool = False,
        keep_in_memory: bool = False,
    ) -> "FakeHFDataset":
        if not batched or input_columns is None:
            raise AssertionError("load_texts must filter batched, on one column")
        mask = function([row[input_columns] for row in self])
        kept = [row for row, keep in zip(self, mask, strict=True) if keep]
        return FakeHFDataset(kept)


@pytest.fixture
def fake_dataset(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch ``load_dataset`` to return the given rows."""

    def _install(rows: list[dict[str, Any]]) -> None:
        monkeypatch.setattr(
            loader_mod, "load_dataset", lambda *a, **k: FakeHFDataset(rows)
        )

    return _install


def test_full_view_uses_whole_conversation_no_gen_prompt(fake_dataset: Any) -> None:
    fake_dataset([{"conversation": CONV}])
    cfg = DatasetConfig(path="x", format="chat", prompt_view="full")
    texts = loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
    assert texts == ["<user>U0<assistant>A0<user>U1<assistant>A1"]


def test_user_view_takes_first_turn_with_gen_prompt(fake_dataset: Any) -> None:
    fake_dataset([{"conversation": CONV}])
    cfg = DatasetConfig(path="x", format="chat", prompt_view="user")
    texts = loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
    # only the first user turn, no assistant text, with the generation prompt.
    assert texts == ["<user>U0<gen>"]


def test_user_view_dedup_collapses_identical_prompts(fake_dataset: Any) -> None:
    fake_dataset([{"conversation": CONV}, {"conversation": CONV}])
    cfg = DatasetConfig(path="x", format="chat", prompt_view="user", dedup=True)
    texts = loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
    assert texts == ["<user>U0<gen>"]


def test_user_view_skips_non_user_opening_turn(fake_dataset: Any) -> None:
    conv = [{"role": "assistant", "content": "A"}, {"role": "user", "content": "U"}]
    fake_dataset([{"conversation": conv}])
    cfg = DatasetConfig(path="x", format="chat", prompt_view="user")
    texts = loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
    assert texts == []


def _origin_rows() -> list[dict[str, Any]]:
    """Three rows from two origins, mirroring guard-glp-benign's schema."""
    return [
        {"conversation": [{"role": "user", "content": "W0"}], "origin": "wildchat_4m"},
        {"conversation": [{"role": "user", "content": "L0"}], "origin": "lmsys"},
        {"conversation": [{"role": "user", "content": "W1"}], "origin": "wildchat_4m"},
    ]


def test_filter_equals_keeps_only_matching_origin(fake_dataset: Any) -> None:
    fake_dataset(_origin_rows())
    cfg = DatasetConfig(
        path="x",
        format="chat",
        prompt_view="user",
        filters=[FilterConfig(column="origin", equals="wildchat_4m")],
    )
    texts = loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
    assert texts == ["<user>W0<gen>", "<user>W1<gen>"]


def test_filter_isin_keeps_any_listed_value(fake_dataset: Any) -> None:
    fake_dataset(_origin_rows())
    cfg = DatasetConfig(
        path="x",
        format="chat",
        prompt_view="user",
        filters=[FilterConfig(column="origin", isin=["lmsys", "wildchat_4m"])],
    )
    texts = loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
    assert texts == ["<user>W0<gen>", "<user>L0<gen>", "<user>W1<gen>"]


def test_stacked_filters_are_conjunctive(fake_dataset: Any) -> None:
    rows = [dict(row, lang="en") for row in _origin_rows()]
    rows[0]["lang"] = "fr"
    fake_dataset(rows)
    cfg = DatasetConfig(
        path="x",
        format="chat",
        prompt_view="user",
        filters=[
            FilterConfig(column="origin", equals="wildchat_4m"),
            FilterConfig(column="lang", equals="en"),
        ],
    )
    texts = loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
    assert texts == ["<user>W1<gen>"]


def test_filter_matching_nothing_yields_no_texts(fake_dataset: Any) -> None:
    fake_dataset(_origin_rows())
    cfg = DatasetConfig(
        path="x",
        format="chat",
        prompt_view="user",
        filters=[FilterConfig(column="origin", equals="wildguard")],
    )
    assert loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1) == []


def test_unknown_prompt_view_raises(fake_dataset: Any) -> None:
    fake_dataset([{"conversation": CONV}])
    cfg = DatasetConfig(path="x", format="chat", prompt_view="bogus")
    with pytest.raises(ValueError, match="unknown prompt_view"):
        loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
