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

import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset

logger = logging.getLogger(__name__)

EVAL_DATASETS = ("guard_glp_data", "wildjailbreak_vanilla")

# fraction of each class routed to calibration (the rest is the test set)
_CALIBRATION_FRACTION = 0.30
# cap on the benign tail reserved for the DTE reference set
_MAX_REFERENCE = 2048


@dataclass
class EvalPrompts:
    """Benign (``*_good``) and adversarial (``*_bad``) prompts for detection.

    ``train_good`` benign prompts are only used to build a DTE reference set;
    ``calibration_*`` pick the decision threshold; ``test_*`` are held out for the
    reported metrics. Positive class = adversarial.
    """

    train_good: list[str]
    calibration_good: list[str]
    calibration_bad: list[str]
    test_good: list[str]
    test_bad: list[str]


def _stratified_split(lst: list[str], cal_frac: float) -> tuple[list[str], list[str]]:
    """Split a (pre-shuffled) list into (calibration, test) by fraction."""
    n_cal = max(1, int(len(lst) * cal_frac))
    return lst[:n_cal], lst[n_cal:]


def _load_guard_glp_data() -> EvalPrompts:
    train: Any = load_dataset("ddidacus/guard-glp-data", split="train")
    calibration: Any = load_dataset("ddidacus/guard-glp-data", split="calibration")
    test: Any = load_dataset("ddidacus/guard-glp-data", split="test")
    return EvalPrompts(
        train_good=[s["prompt"] for s in train if not s["adversarial"]],
        calibration_good=[s["prompt"] for s in calibration if not s["adversarial"]],
        calibration_bad=[s["prompt"] for s in calibration if s["adversarial"]],
        test_good=[s["prompt"] for s in test if not s["adversarial"]],
        test_bad=[s["prompt"] for s in test if s["adversarial"]],
    )


def _load_wildjailbreak_vanilla(seed: int) -> EvalPrompts:
    # Gated TSV, one flat "train" split. Read every column as a string with NA
    # detection off (AllenAI's prescribed invocation): the default type inference
    # otherwise guesses a numeric column and pyarrow aborts when a prompt lands in
    # it, and NA parsing would turn empty cells into NaN floats.
    wjb: Any = load_dataset(
        "allenai/wildjailbreak",
        "train",
        delimiter="\t",
        keep_in_memory=True,
        split="train",
        dtype="str",
        keep_default_na=False,
    )
    cols = wjb.column_names
    # Fail loudly with the real schema if our column assumptions are wrong, so a
    # single run tells us exactly what to rename rather than raising a cryptic
    # KeyError deep in a comprehension.
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
        r["vanilla"] for r in wjb if r["data_type"] == "vanilla_benign" and r["vanilla"]
    ]
    harmful = [
        r["vanilla"]
        for r in wjb
        if r["data_type"] == "vanilla_harmful" and r["vanilla"]
    ]
    if not benign or not harmful:
        raise ValueError(
            "wildjailbreak: found no vanilla_benign/vanilla_harmful rows. "
            f"data_type counts: {dict(Counter(wjb['data_type']))}"
        )
    logger.info(
        "wildjailbreak vanilla: %d benign, %d harmful", len(benign), len(harmful)
    )

    # deterministic seeded shuffle, then per-class stratified cal/test split.
    # (noqa S311: this is a reproducible data split, not a security context.)
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(benign)
    rng.shuffle(harmful)

    calibration_bad, test_bad = _stratified_split(harmful, _CALIBRATION_FRACTION)
    # hold out a benign tail for the DTE reference before splitting cal/test
    n_ref = min(_MAX_REFERENCE, len(benign) // 4)
    train_good = benign[:n_ref]
    calibration_good, test_good = _stratified_split(
        benign[n_ref:], _CALIBRATION_FRACTION
    )
    return EvalPrompts(
        train_good=train_good,
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
