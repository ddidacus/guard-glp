"""CPU-only tests for the detection eval prompt loader (no network).

``load_eval_prompts`` is exercised with a monkeypatched ``load_dataset`` so no
dataset download is needed. Covers source dispatch and the WildJailbreak vanilla
path: label mapping (vanilla_harmful -> bad, vanilla_benign -> good), the
deterministic seeded split, and the loud schema/empty-class failures.
"""

from typing import Any

import pytest

from glp.dataset import eval_prompts as ep_mod
from glp.dataset import load_eval_prompts


class _FakeWJB(list):  # type: ignore[type-arg]
    """A minimal stand-in for an HF Dataset: iterable of row dicts plus the
    ``column_names`` attribute and column access (``ds['col']``) the loader uses."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        super().__init__(rows)
        self.column_names = list(rows[0].keys()) if rows else []

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


def _wjb_rows(n_benign: int, n_harmful: int) -> list[dict[str, Any]]:
    rows = [
        {"vanilla": f"benign {i}", "adversarial": "", "data_type": "vanilla_benign"}
        for i in range(n_benign)
    ]
    rows += [
        {"vanilla": f"harmful {i}", "adversarial": "", "data_type": "vanilla_harmful"}
        for i in range(n_harmful)
    ]
    # an adversarial_* row that must be ignored by the vanilla loader
    rows.append(
        {"vanilla": "", "adversarial": "jailbreak", "data_type": "adversarial_harmful"}
    )
    return rows


@pytest.fixture
def fake_wjb(monkeypatch: pytest.MonkeyPatch):
    def _install(rows: list[dict[str, Any]]) -> None:
        monkeypatch.setattr(
            ep_mod, "load_dataset", lambda *a, **k: _FakeWJB(rows)
        )

    return _install


def test_unknown_dataset_raises() -> None:
    with pytest.raises(NotImplementedError):
        load_eval_prompts("does_not_exist")


def test_wildjailbreak_maps_labels_and_ignores_adversarial(fake_wjb: Any) -> None:
    fake_wjb(_wjb_rows(n_benign=200, n_harmful=100))
    prompts = load_eval_prompts("wildjailbreak_vanilla")

    # every adversarial prompt is a vanilla_harmful one; no adversarial_* leaks in
    all_bad = prompts.calibration_bad + prompts.test_bad
    assert all_bad and all(p.startswith("harmful") for p in all_bad)
    all_good = prompts.train_good + prompts.calibration_good + prompts.test_good
    assert all_good and all(p.startswith("benign") for p in all_good)
    # 100 harmful in -> 100 split across cal/test (nothing dropped)
    assert len(all_bad) == 100
    # benign are partitioned into train_good tail + cal + test, no duplication
    assert len(all_good) == 200
    assert len(set(all_good)) == 200


def test_wildjailbreak_split_is_deterministic(fake_wjb: Any) -> None:
    rows = _wjb_rows(n_benign=200, n_harmful=100)
    fake_wjb(rows)
    a = load_eval_prompts("wildjailbreak_vanilla", seed=123)
    fake_wjb(rows)
    b = load_eval_prompts("wildjailbreak_vanilla", seed=123)
    assert a == b

    fake_wjb(rows)
    c = load_eval_prompts("wildjailbreak_vanilla", seed=999)
    # a different seed yields a different split
    assert c.test_bad != a.test_bad


def test_wildjailbreak_missing_column_raises_loudly(fake_wjb: Any) -> None:
    fake_wjb([{"prompt": "x", "label": "harmful"}])  # wrong schema
    with pytest.raises(KeyError, match="data_type"):
        load_eval_prompts("wildjailbreak_vanilla")


def test_wildjailbreak_empty_class_raises(fake_wjb: Any) -> None:
    fake_wjb(_wjb_rows(n_benign=10, n_harmful=0))  # no harmful rows
    with pytest.raises(ValueError, match="vanilla_benign/vanilla_harmful"):
        load_eval_prompts("wildjailbreak_vanilla")
