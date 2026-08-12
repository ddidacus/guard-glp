"""In-distribution "whitelist" + out-of-distribution prompt pools for OOD detection.

This models detection as an **OOD** problem, not benign-vs-harmful semantics: the GLP's
training distribution is a global in-distribution (ID) *whitelist* (guard-glp-benign,
WildChat portion), and every other prompt source — jailbreaks and plain-harmful alike —
is out-of-distribution and should be flagged. The ID pool is the shared negative class
across every task/regime; each OOD pool is a positive class.

Each loader returns a ``PromptPool`` with a deterministic, seeded, deduplicated
``train`` / ``cal`` / ``test`` split. The ID split is loaded once and reused everywhere,
so the whitelist is identical across all comparisons. Supervised baselines train on
ID-``train`` (neg) vs a regime's OOD-``train`` (pos); training-free GLP scores ID-``test``
(neg) vs each OOD-``test`` (pos).

Prompts are returned as RAW text (ID = first user-turn content; OOD = the instruction
string). Chat-template wrapping to the useronly view is applied uniformly at extraction
time, not here, so this module stays LLM/tokenizer-agnostic.
"""

import csv
import hashlib
import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# per-pool train/cal/test fractions (sum to 1.0). Train feeds supervised baselines;
# cal picks the threshold; test is the held-out reported set.
_TRAIN_FRACTION = 0.50
_CALIBRATION_FRACTION = 0.10

# default per-split caps so no single large OOD set dominates (balanced budget).
# Applied AFTER the split, per split, so small sets just use what they have.
_MAX_TRAIN = 1000
_MAX_CAL = 250
_MAX_TEST = 500

# guard-glp-benign split used for the ID pool. The repo may only ship a `train`
# split (the finalize holdout is tiny); we stream it and hold out our own test subset,
# so streaming from `train` is fine and avoids downloading all ~71 shards.
_ID_SPLIT = "train"

# OOD sets grouped by category (positive/OOD class only).
OOD_JAILBREAK = ("harmbench_gcg", "wjb_vanilla", "wjb_adversarial")
OOD_HARMFUL = ("advbench", "harmbench", "toxicchat")
OOD_SETS = OOD_JAILBREAK + OOD_HARMFUL


@dataclass
class PromptPool:
    """A deduplicated, seeded train/cal/test split of one prompt source (raw text)."""

    train: list[str]
    cal: list[str]
    test: list[str]


def _dedup(texts: list[Any]) -> list[str]:
    # values come straight from HF datasets (untyped) — filter non-str/empty/None
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        h = hashlib.sha256(t.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(t)
    return out


def _make_pool(texts: list[str], seed: int) -> PromptPool:
    """Dedup, seeded-shuffle, split train/cal/test, then cap each split."""
    texts = _dedup(texts)
    rng = random.Random(seed)  # noqa: S311 - reproducible split, not security
    rng.shuffle(texts)
    n = len(texts)
    n_train = int(n * _TRAIN_FRACTION)
    n_cal = max(1, int(n * _CALIBRATION_FRACTION))
    train = texts[:n_train][:_MAX_TRAIN]
    cal = texts[n_train : n_train + n_cal][:_MAX_CAL]
    test = texts[n_train + n_cal :][:_MAX_TEST]
    return PromptPool(train=train, cal=cal, test=test)


# ── In-distribution whitelist ────────────────────────────────────────────────


def load_id_pool(seed: int = 42) -> PromptPool:
    """guard-glp-benign, test split, WildChat portion — the shared ID whitelist.

    Rows are ``{conversation, origin}`` (benign-only, no ``prompt`` field). We keep
    ``origin == 'wildchat_4m'`` and take the first user turn's content as the raw
    prompt (chat-template wrapping happens at extraction).

    The dataset ships as ~71 large parquet shards; a plain ``load_dataset(split=...)``
    would materialize all of them (tens of GB). We **stream** instead and stop once we
    have enough wildchat prompts for the pool (train+cal+test caps), so only a few
    shards are ever fetched.
    """
    need = _MAX_TRAIN + _MAX_CAL + _MAX_TEST
    # over-collect a bit so the seeded split has slack, but stay far below full download
    target = int(need / (_TRAIN_FRACTION + _CALIBRATION_FRACTION) * 1.2) + need
    ds: Any = load_dataset(
        "ddidacus/guard-glp-benign", split=_ID_SPLIT, streaming=True
    )
    texts: list[str] = []
    origins: Counter[Any] = Counter()
    for row in ds:
        origins[row.get("origin")] += 1
        if row.get("origin") != "wildchat_4m":
            continue
        conv = row.get("conversation")
        if not conv or conv[0].get("role") != "user":
            continue
        content = conv[0].get("content")
        if isinstance(content, str) and content:
            texts.append(content)
            if len(texts) >= target:
                break
    if not texts:
        raise ValueError(
            "guard-glp-benign: no wildchat_4m rows with a leading user turn found "
            f"(origins seen: {dict(origins)})."
        )
    logger.info("ID pool (guard-glp-benign wildchat, streamed): %d prompts", len(texts))
    return _make_pool(texts, seed)


# ── Out-of-distribution pools (positive class) ───────────────────────────────


def _require_col(cols: list[str], want: str, name: str) -> None:
    if want not in cols:
        raise KeyError(
            f"{name}: expected a {want!r} column; got {cols}. "
            "Update glp.dataset.ood_prompts to match the dataset schema."
        )


def _load_advbench(seed: int) -> PromptPool:
    ds: Any = load_dataset("walledai/AdvBench", split="train")
    _require_col(ds.column_names, "prompt", "advbench")
    return _make_pool([r["prompt"] for r in ds], seed)


def _load_harmbench(seed: int) -> PromptPool:
    # HarmBench standard behaviors. The 'standard' config lists harmful behaviors; the
    # instruction text is the 'Behavior' column (walledai mirror uses 'prompt').
    ds: Any = load_dataset("walledai/HarmBench", "standard", split="train")
    cols = ds.column_names
    field = "prompt" if "prompt" in cols else ("Behavior" if "Behavior" in cols else None)
    if field is None:
        raise KeyError(
            f"harmbench: expected a 'prompt' or 'Behavior' column; got {cols}. "
            "Update glp.dataset.ood_prompts to match the dataset schema."
        )
    return _make_pool([r[field] for r in ds], seed)


def _load_harmbench_gcg(seed: int) -> PromptPool:
    # Procedurally-generated GCG jailbreaks: HarmBench test cases with the GCG adversarial
    # suffix appended. Sourced from the HarmBench attack artifacts; the full jailbreak
    # string is the 'test_case' (fallback 'prompt'/'jailbreak') column.
    ds: Any = load_dataset("walledai/HarmBench-gcg", split="train")
    cols = ds.column_names
    field = next((c for c in ("test_case", "jailbreak", "prompt") if c in cols), None)
    if field is None:
        raise KeyError(
            f"harmbench_gcg: expected 'test_case'/'jailbreak'/'prompt'; got {cols}. "
            "Update glp.dataset.ood_prompts to match the dataset schema."
        )
    return _make_pool([r[field] for r in ds], seed)


def _read_wildjailbreak_tsv() -> pd.DataFrame:
    # gated TSV; parse directly with pandas (QUOTE_NONE + skip malformed) — the HF csv
    # builder mis-parses the many literal quotes/tabs in these prompts.
    tsv_path = hf_hub_download(
        "allenai/wildjailbreak", "train/train.tsv", repo_type="dataset"
    )
    return pd.read_csv(
        tsv_path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        dtype=str,
        on_bad_lines="skip",
    )


def _load_wjb(column: str, data_type: str, name: str, seed: int) -> PromptPool:
    df = _read_wildjailbreak_tsv()
    cols = list(df.columns)
    _require_col(cols, "data_type", name)
    _require_col(cols, column, name)
    texts = [
        p
        for p, dt in zip(df[column], df["data_type"], strict=True)
        if dt == data_type and p
    ]
    if not texts:
        raise ValueError(
            f"{name}: no {data_type} rows. data_type counts: "
            f"{dict(Counter(df['data_type']))}"
        )
    return _make_pool(texts, seed)


def _load_wjb_vanilla(seed: int) -> PromptPool:
    # plain harmful prompts (no jailbreak wrapper)
    return _load_wjb("vanilla", "vanilla_harmful", "wjb_vanilla", seed)


def _load_wjb_adversarial(seed: int) -> PromptPool:
    # LLM-paraphrased jailbreak-wrapped harmful prompts
    return _load_wjb("adversarial", "adversarial_harmful", "wjb_adversarial", seed)


def _load_toxicchat(seed: int) -> PromptPool:
    # real user prompts flagged toxic. Config 'toxicchat0124'; toxic flag in 'toxicity'.
    ds: Any = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    cols = ds.column_names
    _require_col(cols, "user_input", "toxicchat")
    _require_col(cols, "toxicity", "toxicchat")
    texts = [r["user_input"] for r in ds if int(r["toxicity"]) == 1]
    if not texts:
        raise ValueError("toxicchat: no rows with toxicity==1 found.")
    return _make_pool(texts, seed)


# registry: name -> loader(seed) -> PromptPool
_OOD_LOADERS = {
    "advbench": _load_advbench,
    "harmbench": _load_harmbench,
    "harmbench_gcg": _load_harmbench_gcg,
    "wjb_vanilla": _load_wjb_vanilla,
    "wjb_adversarial": _load_wjb_adversarial,
    "toxicchat": _load_toxicchat,
}


def ood_pool_names() -> tuple[str, ...]:
    """Names of the registered OOD pools (should equal :data:`OOD_SETS`)."""
    return tuple(_OOD_LOADERS)


def load_ood_pool(name: str, seed: int = 42) -> PromptPool:
    """Load one OOD prompt pool by name (see :data:`OOD_SETS`)."""
    if name not in _OOD_LOADERS:
        raise NotImplementedError(
            f"OOD set {name!r} not implemented yet; available: {ood_pool_names()}"
        )
    return _OOD_LOADERS[name](seed)


# ── Task assembly (ID whitelist vs. one OOD pool) ────────────────────────────

# task-name prefix used in eval configs (dataset: "ood:advbench")
OOD_TASK_PREFIX = "ood:"


@dataclass
class OODTask:
    """One ID-vs-OOD task: benign = ID whitelist, adversarial = one OOD pool.

    ``train_good``/``train_bad`` feed supervised baselines; ``cal_*`` pick the
    threshold; ``test_*`` are the held-out reported set. All raw text (chat-template
    wrapping is applied at extraction).
    """

    train_good: list[str]
    train_bad: list[str]
    cal_good: list[str]
    cal_bad: list[str]
    test_good: list[str]
    test_bad: list[str]


def load_ood_task(name: str, seed: int = 42) -> OODTask:
    """Assemble the ID whitelist (negatives) against one OOD pool (positives).

    ``name`` is the OOD set name, optionally with the ``ood:`` prefix used in configs.
    The ID pool is loaded once and reused across every task/regime.
    """
    ood_name = name[len(OOD_TASK_PREFIX) :] if name.startswith(OOD_TASK_PREFIX) else name
    idp = load_id_pool(seed)
    oodp = load_ood_pool(ood_name, seed)
    return OODTask(
        train_good=idp.train,
        train_bad=oodp.train,
        cal_good=idp.cal,
        cal_bad=oodp.cal,
        test_good=idp.test,
        test_bad=oodp.test,
    )


def is_ood_task(dataset: str) -> bool:
    """True if ``dataset`` names an ID-vs-OOD task (``ood:<name>``)."""
    return dataset.startswith(OOD_TASK_PREFIX)


def chat_wrap(texts: list[str], tokenizer: Any) -> list[str]:
    """Render each raw prompt as a single-user-turn chat-template string.

    Matches the ``useronly`` training view of the GLP: a lone user turn with the
    assistant generation prompt appended. Applied uniformly to ID and OOD prompts at
    extraction time so activations match the GLP's training distribution.
    """
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for t in texts
    ]


__all__ = [
    "OOD_HARMFUL",
    "OOD_JAILBREAK",
    "OOD_SETS",
    "OOD_TASK_PREFIX",
    "OODTask",
    "PromptPool",
    "chat_wrap",
    "is_ood_task",
    "load_id_pool",
    "load_ood_pool",
    "load_ood_task",
    "ood_pool_names",
]
