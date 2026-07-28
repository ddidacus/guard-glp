"""Cross-dataset generalization for the supervised baselines.

Trains a detector on one dataset and evaluates it on another, to test whether the
near-perfect in-distribution scores of the linear probe (and diff-of-means) reflect a
transferable notion of "adversarialness" or just overfitting to dataset-specific
surface cues. If OOD AUROC collapses toward ~0.5, it is the latter.

Both methods reuse the SHARED activation cache (mean-pooled, layer 14, per llm), so no
new extraction happens if the baseline runs (run_baselines_comparison.sh) have already
populated it — this is a pass-2-only, CPU-capable computation.

    python eval_cross_dataset.py \
        --train_dataset=guard_glp_data --test_dataset=wildjailbreak_vanilla \
        --out_dir=results/xeval-lp-gg2wjb --method=probe

`method` in {probe, diffmean}. Prints AUPRC/AUROC and writes results.json.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fire
import numpy as np
import numpy.typing as npt
import torch
from eval_linear_probe import _score_probe, _train_probe
from evaluate_classifier import _classification_metrics, _find_best_f1_threshold

from glp.dataset import cached_activations
from glp.dataset.eval_prompts import EVAL_DATASETS

NDArray = npt.NDArray[Any]


def _score_diffmean(acts: torch.Tensor, sv: NDArray) -> NDArray:
    """dot(acts, sv) / norm(acts) — higher = more adversarial (as in eval_diffmean)."""
    sv_t = torch.from_numpy(sv).float()
    a = acts.float()
    return (a @ sv_t / (a.norm(dim=1) + 1e-8)).numpy()


def _diffmean_scorer(sv: NDArray) -> Callable[[torch.Tensor], NDArray]:
    return lambda a: _score_diffmean(a, sv)


def _probe_scorer(
    probe: Any, device: str, batch_size: int
) -> Callable[[torch.Tensor], NDArray]:
    # _score_probe returns P(benign); adversarial score = 1 - that
    return lambda a: 1.0 - _score_probe(probe, a, device, batch_size)

_DEFAULT_LLM = "meta-llama/Llama-3.2-1B-Instruct"


def _split_acts(
    dataset: str, split: str, llm_model_id: str, layers: list[int], pooling: str
) -> torch.Tensor:
    """(N, L, D) activations for one split, from the shared cache.

    ``extract`` raises: this script is pass-2 only and must not silently re-extract
    (which would need a GPU + the LLM). Populate the cache via the baseline runs first.
    """

    def _no_extract() -> torch.Tensor:
        raise RuntimeError(
            f"activation cache miss for {dataset}/{split} "
            f"(llm={llm_model_id}, layers={layers}, pooling={pooling}). "
            "Run the baseline extraction first so the shared cache is populated."
        )

    return cached_activations(
        dataset=dataset,
        llm_model_id=llm_model_id,
        layers=layers,
        token_pooling=pooling,
        split=split,
        shard=0,
        extract=_no_extract,
    )


def main(
    train_dataset: str,
    test_dataset: str,
    out_dir: str,
    method: str = "probe",
    layers: list[int] | None = None,
    llm_model_id: str = _DEFAULT_LLM,
    token_pooling: str = "mean",
    probe_lr: float = 1e-3,
    probe_epochs: int = 100,
    probe_wd: float = 1e-4,
    probe_batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    layers = layers or [14]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # fail fast on an unknown dataset name (without loading — this is pass-2 only)
    for ds in (train_dataset, test_dataset):
        if ds not in EVAL_DATASETS:
            raise ValueError(f"unknown dataset {ds!r}; expected one of {EVAL_DATASETS}")

    def acts(ds: str, split: str) -> torch.Tensor:
        return _split_acts(ds, split, llm_model_id, layers, token_pooling)

    # training data comes from `train_dataset`, evaluation from `test_dataset`
    train_good = acts(train_dataset, "train_good")
    train_bad = acts(train_dataset, "train_bad")
    # threshold is calibrated on the TEST dataset's calibration split (a detector
    # deployed on B would tune its threshold on B), test on B's test split
    cal_good = acts(test_dataset, "cal_good")
    cal_bad = acts(test_dataset, "cal_bad")
    test_good = acts(test_dataset, "test_good")
    test_bad = acts(test_dataset, "test_bad")

    print("================================================")
    print(f"[+] method:        {method}")
    print(f"[+] train_dataset: {train_dataset}  (good={len(train_good)} bad={len(train_bad)})")
    print(f"[+] test_dataset:  {test_dataset}  (test good={len(test_good)} bad={len(test_bad)})")
    print(f"[+] layers:        {layers}")
    print("================================================")

    results: dict[str, Any] = {
        "config": {
            "method": method,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "layers": layers,
            "llm_model_id": llm_model_id,
            "token_pooling": token_pooling,
        },
        "per_layer": {},
        "aggregate": {},
    }

    cal_labels = np.concatenate([np.zeros(len(cal_good)), np.ones(len(cal_bad))])
    test_labels = np.concatenate([np.zeros(len(test_good)), np.ones(len(test_bad))])

    if method not in ("probe", "diffmean"):
        raise NotImplementedError(f"Unknown method {method!r} (probe|diffmean)")

    if method not in ("probe", "diffmean"):
        raise NotImplementedError(f"Unknown method {method!r} (probe|diffmean)")

    layer_auprcs: list[tuple[int, float]] = []
    best: dict[str, Any] = {}
    best_auprc = -1.0
    for li, layer in enumerate(layers):
        # fit a per-layer detector on the TRAIN dataset -> a scorer for TEST acts
        if method == "probe":
            probe = _train_probe(
                torch.cat([train_good[:, li, :], train_bad[:, li, :]], dim=0),
                torch.cat([torch.ones(len(train_good)), torch.zeros(len(train_bad))]),
                lr=probe_lr,
                num_epochs=probe_epochs,
                weight_decay=probe_wd,
                batch_size=probe_batch_size,
                device=device,
            )
            score_fn = _probe_scorer(probe, device, probe_batch_size)
        else:  # diffmean
            pos = train_bad[:, li, :].float().mean(0).numpy()
            neg = train_good[:, li, :].float().mean(0).numpy()
            sv = pos - neg
            score_fn = _diffmean_scorer(sv / (np.linalg.norm(sv) + 1e-8))

        cal_scores = np.concatenate(
            [score_fn(cal_good[:, li, :]), score_fn(cal_bad[:, li, :])]
        )
        test_scores = np.concatenate(
            [score_fn(test_good[:, li, :]), score_fn(test_bad[:, li, :])]
        )
        youden = _find_best_f1_threshold(cal_labels, cal_scores)
        m = _classification_metrics(
            test_labels, test_scores, f"layer {layer} {method}", best_f1_threshold=youden
        )
        results["per_layer"][f"layer_{layer}"] = m
        layer_auprcs.append((layer, m["auprc"]))
        if m["auprc"] > best_auprc:
            best_auprc, best = m["auprc"], {**m, "best_layer": layer}

    results["aggregate"]["best_layer"] = best
    out_json = Path(out_dir) / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(
        f"\n[{method}] {train_dataset} -> {test_dataset}   "
        f"best-layer AUPRC={best['auprc']:.4f}  AUROC={best['auroc']:.4f}"
    )
    print(f"Saved to {out_json}")
    return results


if __name__ == "__main__":
    fire.Fire(main)
