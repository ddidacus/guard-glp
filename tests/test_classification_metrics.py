"""Test that _classification_metrics reports both AUPRC and AUROC.

All three detection scripts (recon eval, diff-of-means, linear probe) route their
per-layer metrics through this one function, so both metrics being present here is
what standardizes AUPRC+AUROC across every method.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "detection"))

from evaluate_classifier import _classification_metrics  # noqa: E402


def test_reports_auprc_and_auroc() -> None:
    labels = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.6])
    m = _classification_metrics(labels, scores, "test", verbose=False)
    assert "auprc" in m and "auroc" in m
    # perfectly separable here -> both should be 1.0
    assert m["auprc"] == pytest.approx(1.0)
    assert m["auroc"] == pytest.approx(1.0)


def test_auroc_is_half_for_random_ranking() -> None:
    # interleaved scores give a coin-flip ranking -> AUROC ~ 0.5
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    m = _classification_metrics(labels, scores, "tie", verbose=False)
    assert m["auroc"] == pytest.approx(0.5)
