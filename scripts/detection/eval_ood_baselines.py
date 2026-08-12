"""Supervised OOD-detection baselines (linear probe + diff-of-means) across regimes.

Framing: guard-glp-benign (WildChat) is the in-distribution (ID) whitelist (negatives);
each of the 6 OOD sets is a positive pool. We train detectors in three regimes and
evaluate every detector on every OOD test set (the generalization matrix), to contrast
with the training-free GLPs:

  - per-task     : train ID vs one OOD set            (6 detectors)
  - per-category : train ID vs {3 jailbreak}, {3 harmful}   (2 detectors)
  - all-ood      : train ID vs all 6 OOD combined     (1 detector)

Two passes:
  run       (GPU) — extract & cache activations for ID + all OOD splits, all 16 layers,
                    mean-pooled, chat-template wrapped (the useronly GLP view). Uses the
                    shared cache, so ID is extracted once and reused.
  aggregate (CPU) — train each regime's detectors on cached train acts, evaluate on
                    every OOD test set, write per-layer + best-layer AUPRC/AUROC.

    python eval_ood_baselines.py run --gpu_id=0 [--llm_model_id=...]
    python eval_ood_baselines.py aggregate --out_dir=results/ood/baselines
"""

import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
import numpy.typing as npt
import torch
from eval_linear_probe import _score_probe, _train_probe
from evaluate_classifier import (
    _chunk,
    _classification_metrics,
    _find_best_f1_threshold,
    extract_activations,
)

from glp.dataset import cached_activations, load_id_pool, load_ood_pool
from glp.dataset.ood_prompts import (
    OOD_HARMFUL,
    OOD_JAILBREAK,
    OOD_SETS,
    chat_wrap,
)

NDArray = npt.NDArray[Any]

_DEFAULT_LLM = "meta-llama/Llama-3.2-1B-Instruct"
_LAYERS = list(range(16))
_POOLING = "mean"  # supervised baselines pool over tokens
# regime name -> tuple of OOD set names whose TRAIN pools form the positive class
_REGIMES: dict[str, tuple[str, ...]] = {
    **{f"task:{n}": (n,) for n in OOD_SETS},
    "cat:jailbreak": OOD_JAILBREAK,
    "cat:harmful": OOD_HARMFUL,
    "all_ood": OOD_SETS,
}


# ── Pass 1: extraction (populate the shared cache) ───────────────────────────


def run(
    gpu_id: int = 0,
    llm_model_id: str = _DEFAULT_LLM,
    glp_model_id: str = "generative-latent-prior/glp-llama1b-d12-multi",
    glp_checkpoint: str = "final",
) -> None:
    """Extract & cache activations for ID + all OOD splits (all 16 layers, mean-pooled).

    A GLP is loaded only to reuse its tracedict_config (layer prefix/retain); the
    baselines operate on raw LLM activations. Keyed so ID is extracted once ('id_pool')
    and each OOD under its own name — shared with any other run at the same (llm,layers,
    pooling).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from glp.denoiser import load_glp

    device = f"cuda:{gpu_id}"
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    glp = load_glp(glp_model_id, device=device, checkpoint=glp_checkpoint)
    from typing import cast

    cast(Any, glp.tracedict_config).layers = _LAYERS

    def _extract_split(cache_ds: str, split: str, texts: list[str]) -> None:
        def _do() -> torch.Tensor:
            wrapped = chat_wrap(texts, llm_tokenizer)
            return torch.cat(
                [
                    extract_activations(
                        b,
                        llm_model,
                        llm_tokenizer,
                        glp,
                        device=device,
                        batch_size=64,
                        token_pooling=_POOLING,
                    ).cpu()
                    for b in _chunk(wrapped, 64)
                ]
            )

        acts = cached_activations(
            dataset=cache_ds,
            llm_model_id=llm_model_id,
            layers=_LAYERS,
            token_pooling=_POOLING,
            split=split,
            shard=0,
            extract=_do,
        )
        print(f"  {cache_ds}/{split}: {tuple(acts.shape)}")

    print("[+] Extracting ID pool (once)...")
    idp = load_id_pool()
    for split, texts in (("train", idp.train), ("cal", idp.cal), ("test", idp.test)):
        _extract_split("id_pool", split, texts)

    for name in OOD_SETS:
        print(f"[+] Extracting OOD '{name}'...")
        oodp = load_ood_pool(name)
        for split, texts in (
            ("train", oodp.train),
            ("cal", oodp.cal),
            ("test", oodp.test),
        ):
            _extract_split(name, split, texts)
    print("[+] Extraction complete.")


# ── Pass 2: train regimes + evaluate ─────────────────────────────────────────


def _acts(cache_ds: str, split: str, llm_model_id: str) -> torch.Tensor:
    def _miss() -> torch.Tensor:
        raise RuntimeError(
            f"activation cache miss for {cache_ds}/{split}. Run the `run` pass first."
        )

    return cached_activations(
        dataset=cache_ds,
        llm_model_id=llm_model_id,
        layers=_LAYERS,
        token_pooling=_POOLING,
        split=split,
        shard=0,
        extract=_miss,
    )


def _score_layer(
    method: str,
    train_neg: torch.Tensor,
    train_pos: torch.Tensor,
    a: torch.Tensor,
    probe_kwargs: dict[str, Any],
) -> NDArray:
    """Fit method on (neg,pos) train acts for one layer, return OOD scores for `a`."""
    if method == "probe":
        x = torch.cat([train_neg, train_pos], dim=0)
        y = torch.cat([torch.ones(len(train_neg)), torch.zeros(len(train_pos))])
        probe = _train_probe(x, y, **probe_kwargs)
        # _score_probe returns P(benign); OOD score = 1 - that
        return 1.0 - _score_probe(probe, a, probe_kwargs["device"], probe_kwargs["batch_size"])
    # diffmean
    sv = train_pos.float().mean(0).numpy() - train_neg.float().mean(0).numpy()
    sv = sv / (np.linalg.norm(sv) + 1e-8)
    return (a.float() @ torch.from_numpy(sv).float()).numpy()


def _balance_rows(
    neg: torch.Tensor, n_pos: int, id_ratio: float, seed: int
) -> torch.Tensor:
    """Seeded downsample of negative activation rows to id_ratio * n_pos (keep all pos).

    Mirrors glp.dataset.ood_prompts._balance at the activation-row level, so the
    baselines evaluate/train on the same class balance the GLP recon uses.
    """
    target = min(len(neg), max(1, int(round(id_ratio * n_pos))))
    if target >= len(neg):
        return neg
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(neg), generator=g)[:target]
    return neg[idx]


def aggregate(
    out_dir: str = "results/ood/baselines",
    llm_model_id: str = _DEFAULT_LLM,
    id_ratio: float = 1.0,
    probe_lr: float = 1e-3,
    probe_epochs: int = 100,
    probe_wd: float = 1e-4,
    probe_batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    probe_kwargs = {
        "lr": probe_lr,
        "num_epochs": probe_epochs,
        "weight_decay": probe_wd,
        "batch_size": probe_batch_size,
        "device": device,
    }

    # cached activations (N, 16, D)
    id_train = _acts("id_pool", "train", llm_model_id)
    id_test = _acts("id_pool", "test", llm_model_id)
    ood_train = {n: _acts(n, "train", llm_model_id) for n in OOD_SETS}
    ood_test = {n: _acts(n, "test", llm_model_id) for n in OOD_SETS}

    results: dict[str, Any] = {"config": {"llm_model_id": llm_model_id}, "regimes": {}}

    for method in ("probe", "diffmean"):
        for regime, pos_sets in _REGIMES.items():
            # training: ID negatives balanced to id_ratio * combined-OOD-train size
            pos_train_all = torch.cat([ood_train[n] for n in pos_sets], dim=0)
            neg_train = _balance_rows(id_train, len(pos_train_all), id_ratio, seed=42)

            row: dict[str, Any] = {}
            for eval_name in OOD_SETS:
                # per eval-set: ID-test balanced to id_ratio * this OOD's test size,
                # matching the recon eval's balanced test set (seed 44 == 42 + 2)
                pos_test = ood_test[eval_name]
                neg_test = _balance_rows(id_test, len(pos_test), id_ratio, seed=44)
                best_auprc, best = -1.0, {}
                for li in _LAYERS:
                    scores = _score_layer(
                        method,
                        neg_train[:, li, :],
                        pos_train_all[:, li, :],
                        torch.cat([neg_test[:, li, :], pos_test[:, li, :]], dim=0),
                        probe_kwargs,
                    )
                    labels = np.concatenate(
                        [np.zeros(len(neg_test)), np.ones(len(pos_test))]
                    )
                    thr = _find_best_f1_threshold(labels, scores)
                    m = _classification_metrics(
                        labels, scores, f"{method}/{regime}->{eval_name} L{li}",
                        verbose=False, best_f1_threshold=thr,
                    )
                    if m["auprc"] > best_auprc:
                        best_auprc, best = m["auprc"], {**m, "best_layer": li}
                row[eval_name] = best
                print(
                    f"  {method:8s} {regime:16s} -> {eval_name:16s}  "
                    f"L{best['best_layer']:<2d} AUPRC={best['auprc']:.4f} "
                    f"AUROC={best['auroc']:.4f}"
                )
            results["regimes"].setdefault(method, {})[regime] = row

    out_json = Path(out_dir) / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_json}")
    return results


if __name__ == "__main__":
    fire.Fire({"run": run, "aggregate": aggregate})
