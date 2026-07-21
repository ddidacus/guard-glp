"""CPU-only tests for the text loader's ``prompt_view`` handling (no network).

``load_texts`` is exercised with a fake tokenizer and a monkeypatched
``load_dataset`` so no model or dataset download is needed. Covers the
``full`` vs ``user`` chat views and the defensive skip of non-user opening turns.
"""

from typing import Any

import pytest

from glp.dataset import loader as loader_mod
from glp.dataset.builder import DatasetConfig


class FakeTokenizer:
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


CONV = [
    {"role": "user", "content": "U0"},
    {"role": "assistant", "content": "A0"},
    {"role": "user", "content": "U1"},
    {"role": "assistant", "content": "A1"},
]


@pytest.fixture
def fake_dataset(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch ``load_dataset`` to return the given rows (with a no-op filter)."""

    def _install(rows: list[dict[str, Any]]) -> None:
        class FakeHFDataset(list):  # type: ignore[type-arg]
            def filter(self, *args: Any, **kwargs: Any) -> "FakeHFDataset":
                return self

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


def test_unknown_prompt_view_raises(fake_dataset: Any) -> None:
    fake_dataset([{"conversation": CONV}])
    cfg = DatasetConfig(path="x", format="chat", prompt_view="bogus")
    with pytest.raises(ValueError, match="unknown prompt_view"):
        loader_mod.load_texts(cfg, FakeTokenizer(), gpu_id=0, num_gpus=1)
