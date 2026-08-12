"""CPU-only tests for the ID/OOD prompt pools (no network).

``load_dataset`` is monkeypatched so no download happens. Covers split hygiene
(train/cal/test disjoint, deduped), the ID WildChat filter, task assembly (ID reused
as negatives), and chat_wrap formatting.
"""

from typing import Any

import pytest

from glp.dataset import ood_prompts as op


class _FakeHF(list):  # type: ignore[type-arg]
    """Iterable of row dicts with a column_names attribute (like an HF Dataset)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        self.column_names = list(rows[0].keys()) if rows else []


@pytest.fixture
def patch_load(monkeypatch: pytest.MonkeyPatch):
    def _install(rows: list[dict[str, Any]]) -> None:
        monkeypatch.setattr(op, "load_dataset", lambda *a, **k: _FakeHF(rows))

    return _install


def _id_rows(n: int, other: int = 3) -> list[dict[str, Any]]:
    rows = [
        {"conversation": [{"role": "user", "content": f"wc prompt {i}"}], "origin": "wildchat_4m"}
        for i in range(n)
    ]
    # lmsys rows + a non-user-first row that must be filtered out
    rows += [
        {"conversation": [{"role": "user", "content": f"lm {i}"}], "origin": "lmsys"}
        for i in range(other)
    ]
    rows.append(
        {"conversation": [{"role": "assistant", "content": "hi"}], "origin": "wildchat_4m"}
    )
    return rows


def test_id_pool_filters_wildchat_and_splits(patch_load: Any) -> None:
    patch_load(_id_rows(100))
    pool = op.load_id_pool(seed=0)
    allp = pool.train + pool.cal + pool.test
    # only wildchat_4m user-first prompts, nothing from lmsys or the assistant row
    assert all(p.startswith("wc prompt") for p in allp)
    assert len(allp) == 100
    # splits are disjoint
    assert set(pool.train).isdisjoint(pool.cal)
    assert set(pool.train).isdisjoint(pool.test)
    assert set(pool.cal).isdisjoint(pool.test)


def test_id_pool_dedups(patch_load: Any) -> None:
    rows = [
        {"conversation": [{"role": "user", "content": "dup"}], "origin": "wildchat_4m"}
        for _ in range(20)
    ]
    patch_load(rows)
    pool = op.load_id_pool(seed=0)
    assert pool.train + pool.cal + pool.test == ["dup"]


def test_id_pool_raises_when_no_wildchat(patch_load: Any) -> None:
    patch_load(
        [{"conversation": [{"role": "user", "content": "x"}], "origin": "lmsys"}]
    )
    with pytest.raises(ValueError, match="wildchat_4m"):
        op.load_id_pool()


def test_advbench_pool(patch_load: Any) -> None:
    patch_load([{"prompt": f"harm {i}", "target": "sure"} for i in range(50)])
    pool = op.load_ood_pool("advbench", seed=0)
    allp = pool.train + pool.cal + pool.test
    assert allp and all(p.startswith("harm") for p in allp)
    assert len(allp) == 50


def test_advbench_missing_column_raises(patch_load: Any) -> None:
    patch_load([{"text": "x"}])
    with pytest.raises(KeyError, match="prompt"):
        op.load_ood_pool("advbench")


def test_unknown_ood_pool_raises() -> None:
    with pytest.raises(NotImplementedError):
        op.load_ood_pool("does_not_exist")


def test_all_ood_names_registered() -> None:
    # every active OOD set must have a registered loader (the registry may hold extra,
    # temporarily-disabled loaders like harmbench_gcg)
    assert set(op.OOD_SETS) <= set(op.ood_pool_names())
    assert set(op.OOD_JAILBREAK) | set(op.OOD_HARMFUL) == set(op.OOD_SETS)


def test_toxicchat_filters_toxic(patch_load: Any) -> None:
    rows = [
        {"user_input": f"tox {i}", "toxicity": 1} for i in range(30)
    ] + [{"user_input": f"clean {i}", "toxicity": 0} for i in range(30)]
    patch_load(rows)
    pool = op.load_ood_pool("toxicchat", seed=0)
    allp = pool.train + pool.cal + pool.test
    assert allp and all(p.startswith("tox") for p in allp)


def test_wjb_vanilla_via_tsv(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "vanilla": [f"h{i}" for i in range(20)] + ["", "b1"],
            "adversarial": [""] * 22,
            "data_type": ["vanilla_harmful"] * 20 + ["vanilla_harmful", "vanilla_benign"],
        }
    )
    monkeypatch.setattr(op, "_read_wildjailbreak_tsv", lambda: df)
    pool = op.load_ood_pool("wjb_vanilla", seed=0)
    allp = pool.train + pool.cal + pool.test
    # only non-empty vanilla_harmful, no benign, no empty string
    assert allp and all(p.startswith("h") for p in allp)
    assert len(allp) == 20


def test_load_ood_task_reuses_id_as_negatives(monkeypatch: pytest.MonkeyPatch) -> None:
    id_pool = op.PromptPool(train=["i1"], cal=["i2"], test=["i3"])
    ood_pool = op.PromptPool(train=["o1"], cal=["o2"], test=["o3"])
    monkeypatch.setattr(op, "load_id_pool", lambda seed=42: id_pool)
    monkeypatch.setattr(op, "load_ood_pool", lambda name, seed=42: ood_pool)

    task = op.load_ood_task("ood:advbench")
    assert task.train_good == ["i1"] and task.train_bad == ["o1"]
    assert task.test_good == ["i3"] and task.test_bad == ["o3"]
    assert op.is_ood_task("ood:advbench")
    assert not op.is_ood_task("guard_glp_data")


def test_load_ood_task_balances_50_50(monkeypatch: pytest.MonkeyPatch) -> None:
    # ID much larger than OOD; default id_ratio=1.0 -> equal classes per split
    id_pool = op.PromptPool(
        train=[f"i{i}" for i in range(100)],
        cal=[f"c{i}" for i in range(50)],
        test=[f"t{i}" for i in range(80)],
    )
    ood_pool = op.PromptPool(
        train=[f"o{i}" for i in range(20)],
        cal=[f"oc{i}" for i in range(10)],
        test=[f"ot{i}" for i in range(15)],
    )
    monkeypatch.setattr(op, "load_id_pool", lambda seed=42: id_pool)
    monkeypatch.setattr(op, "load_ood_pool", lambda name, seed=42: ood_pool)

    task = op.load_ood_task("ood:advbench")
    assert len(task.train_good) == len(task.train_bad) == 20
    assert len(task.cal_good) == len(task.cal_bad) == 10
    assert len(task.test_good) == len(task.test_bad) == 15
    # negatives are still ID prompts (downsampled, not fabricated)
    assert all(p.startswith("i") for p in task.train_good)


def test_load_ood_task_id_ratio_2x(monkeypatch: pytest.MonkeyPatch) -> None:
    id_pool = op.PromptPool(train=[f"i{i}" for i in range(100)], cal=["c"], test=["t"])
    ood_pool = op.PromptPool(train=[f"o{i}" for i in range(20)], cal=["oc"], test=["ot"])
    monkeypatch.setattr(op, "load_id_pool", lambda seed=42: id_pool)
    monkeypatch.setattr(op, "load_ood_pool", lambda name, seed=42: ood_pool)

    task = op.load_ood_task("ood:advbench", id_ratio=2.0)
    assert len(task.train_bad) == 20 and len(task.train_good) == 40  # 2x positives


class _FakeTokenizer:
    def apply_chat_template(
        self, conversation: list[dict[str, Any]], tokenize: bool, add_generation_prompt: bool
    ) -> str:
        content = conversation[0]["content"]
        suffix = "<gen>" if add_generation_prompt else ""
        return f"<user>{content}{suffix}"


def test_chat_wrap() -> None:
    out = op.chat_wrap(["hello", "world"], _FakeTokenizer())
    assert out == ["<user>hello<gen>", "<user>world<gen>"]
