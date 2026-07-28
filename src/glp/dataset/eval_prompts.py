"""Labeled prompt sets for the detection eval.

The Guard-GLP detection classifiers score benign vs. adversarial *prompts* and
report AUPRC, picking a decision threshold on a calibration set and evaluating on
a held-out test set. This module owns loading those labeled prompts and shaping
them into the ``(train / calibration / test)`` structure the eval consumes, so
the eval scripts stay free of dataset-specific plumbing.

A source is selected by name (see :data:`EVAL_DATASETS`):

``guard_glp_data``
    ``ddidacus/guard-glp-data`` with its native ``train`` / ``calibration`` /
    ``test`` splits and a boolean ``adversarial`` label (the historical default).

``wildjailbreak_vanilla``
    ``allenai/wildjailbreak`` *vanilla* prompts — ``vanilla_harmful`` as the
    adversarial (positive) class and ``vanilla_benign`` as benign. WildJailbreak
    ships one flat, gated TSV split, so calibration/test are carved from it with a
    deterministic seeded, per-class-stratified split (a benign tail is held out as
    ``train_good`` for DTE reference construction).
"""

import csv
import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

EVAL_DATASETS = ("guard_glp_data", "wildjailbreak_vanilla")

# per-class train/calibration/test fractions for sources without a native split
# (WildJailbreak). Train feeds supervised baselines; cal picks the threshold; test
# is held out. Must sum to 1.0.
_TRAIN_FRACTION = 0.40
_CALIBRATION_FRACTION = 0.20


@dataclass
class EvalPrompts:
    """Benign (``*_good``) and adversarial (``*_bad``) prompts for detection.

    ``train_*`` feed methods that need training data (a DTE reference set uses
    ``train_good``; supervised baselines like the linear probe use both
    ``train_good`` and ``train_bad``); ``calibration_*`` pick the decision
    threshold; ``test_*`` are held out for the reported metrics. Positive class =
    adversarial. ``train_bad`` may be empty for sources/methods that never need it.
    """

    train_good: list[str]
    train_bad: list[str]
    calibration_good: list[str]
    calibration_bad: list[str]
    test_good: list[str]
    test_bad: list[str]


def _three_way_split(lst: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split a (pre-shuffled) list into (train, calibration, test) by fraction."""
    n = len(lst)
    n_train = int(n * _TRAIN_FRACTION)
    n_cal = max(1, int(n * _CALIBRATION_FRACTION))
    train = lst[:n_train]
    calibration = lst[n_train : n_train + n_cal]
    test = lst[n_train + n_cal :]
    return train, calibration, test


def _load_guard_glp_data() -> EvalPrompts:
    train: Any = load_dataset("ddidacus/guard-glp-data", split="train")
    calibration: Any = load_dataset("ddidacus/guard-glp-data", split="calibration")
    test: Any = load_dataset("ddidacus/guard-glp-data", split="test")
    return EvalPrompts(
        train_good=[s["prompt"] for s in train if not s["adversarial"]],
        train_bad=[s["prompt"] for s in train if s["adversarial"]],
        calibration_good=[s["prompt"] for s in calibration if not s["adversarial"]],
        calibration_bad=[s["prompt"] for s in calibration if s["adversarial"]],
        test_good=[s["prompt"] for s in test if not s["adversarial"]],
        test_bad=[s["prompt"] for s in test if s["adversarial"]],
    )


def _load_wildjailbreak_vanilla(seed: int) -> EvalPrompts:
    # Gated TSV, one flat "train" split. The HF csv/datasets builder cannot parse it
    # cleanly: with quoting on, the many literal " in prompts mis-split rows; with
    # quoting off, prompts containing a literal tab explode the field count ("Expected
    # 4 fields, saw 420"). So read the cached TSV directly with pandas instead —
    # QUOTE_NONE (tabs are the only delimiter that matters), NA detection off, and skip
    # the handful of genuinely malformed lines rather than aborting the whole load.
    tsv_path = hf_hub_download(
        "allenai/wildjailbreak",
        "train/train.tsv",
        repo_type="dataset",
    )
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        dtype=str,
        on_bad_lines="skip",
    )
    cols = list(df.columns)
    # Fail loudly with the real schema if our column assumptions are wrong, so a
    # single run tells us exactly what to rename rather than raising a cryptic
    # KeyError.
    if "data_type" not in cols:
        raise KeyError(
            f"wildjailbreak: expected a 'data_type' column; got {cols}. "
            "Update glp.dataset.eval_prompts to match the dataset schema."
        )
    # For vanilla rows the prompt text lives in the 'vanilla' column.
    if "vanilla" not in cols:
        raise KeyError(
            f"wildjailbreak: expected a 'vanilla' prompt column; got {cols}. "
            "Update glp.dataset.eval_prompts to match the dataset schema."
        )

    benign = [
        p
        for p, dt in zip(df["vanilla"], df["data_type"], strict=True)
        if dt == "vanilla_benign" and p
    ]
    harmful = [
        p
        for p, dt in zip(df["vanilla"], df["data_type"], strict=True)
        if dt == "vanilla_harmful" and p
    ]
    if not benign or not harmful:
        raise ValueError(
            "wildjailbreak: found no vanilla_benign/vanilla_harmful rows. "
            f"data_type counts: {dict(Counter(df['data_type']))}"
        )
    logger.info(
        "wildjailbreak vanilla: %d benign, %d harmful", len(benign), len(harmful)
    )

    # deterministic seeded shuffle, then an identical per-class 3-way split so every
    # method has what it needs: training-free GLP scores use only cal/test, while
    # supervised baselines (linear probe) also train on train_good/train_bad.
    # (noqa S311: this is a reproducible data split, not a security context.)
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(benign)
    rng.shuffle(harmful)

    train_good, calibration_good, test_good = _three_way_split(benign)
    train_bad, calibration_bad, test_bad = _three_way_split(harmful)
    return EvalPrompts(
        train_good=train_good,
        train_bad=train_bad,
        calibration_good=calibration_good,
        calibration_bad=calibration_bad,
        test_good=test_good,
        test_bad=test_bad,
    )


def load_eval_prompts(dataset: str = "guard_glp_data", seed: int = 42) -> EvalPrompts:
    """Load labeled detection prompts for ``dataset`` (see :data:`EVAL_DATASETS`).

    ``seed`` drives the deterministic calibration/test split for sources (like
    WildJailbreak) that do not ship one; it is ignored for datasets with native
    splits.
    """
    if dataset == "guard_glp_data":
        return _load_guard_glp_data()
    if dataset == "wildjailbreak_vanilla":
        return _load_wildjailbreak_vanilla(seed)
    raise NotImplementedError(
        f"Unknown eval dataset {dataset!r}; expected one of {EVAL_DATASETS}."
    )
