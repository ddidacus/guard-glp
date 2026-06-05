import json
import os
import random
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from datasets import load_dataset
from eval_linear_probe import _score_probe, _train_probe
from evaluate_classifier import (
    _chunk,
    _classification_metrics,
    _find_best_f1_threshold,
    _make_plots,
    extract_activations,
)
from ood_evaluate_classifier import load_ood_test
from transformers import AutoModelForCausalLM, AutoTokenizer

from glp.denoiser import load_glp

NDArray = npt.NDArray[Any]

# ── Pass 1: activation extraction (per GPU) ──────────────────────────────────


def main(
    gpu_id: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    num_gpus: int = 4,
    token_pooling: str = "mean",
) -> None:
    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

    if model == "1b":
        batch_size = 64
        llm_model_id = "unsloth/Llama-3.2-1B"
        glp_model_id = "generative-latent-prior/glp-llama1b-d12-multi"
    elif model == "8b":
        batch_size = 32
        llm_model_id = "meta-llama/Llama-3.1-8B"
        glp_model_id = "generative-latent-prior/glp-llama8b-d6"
    else:
        raise NotImplementedError(f"Unknown model: {model}")

    print("================================================")
    print("[+] OOD Linear Probe (wildjailbreak adversarial)")
    print(f"[+] LLM:           {llm_model_id}")
    print(f"[+] batch_size:    {batch_size}")
    print(f"[+] layers:        {layers}")
    print(f"[+] gpu:           {gpu_id}/{num_gpus}")
    print(f"[+] token_pooling: {token_pooling}")
    print(f"[+] out_dir:       {out_dir}")
    print("================================================")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(glp_model_id, device=device, checkpoint="final")
    cast(Any, diffusion_model.tracedict_config).layers = layers

    common: dict[str, Any] = {
        "llm_model": llm_model,
        "llm_tokenizer": llm_tokenizer,
        "diffusion_model": diffusion_model,
        "device": device,
        "token_pooling": token_pooling,
    }

    # train set: same in-distribution data
    train_dataset: Any = load_dataset("ddidacus/guard-glp-data", split="train")
    train_good = [s["prompt"] for s in train_dataset if not s["adversarial"]]
    train_bad = [s["prompt"] for s in train_dataset if s["adversarial"]]

    # calibration from in-distribution data, test from OOD wildjailbreak
    calibration_dataset: Any = load_dataset(
        "ddidacus/guard-glp-data", split="calibration"
    )
    cal_good_all = [s["prompt"] for s in calibration_dataset if not s["adversarial"]]
    cal_bad_all = [s["prompt"] for s in calibration_dataset if s["adversarial"]]
    ood = load_ood_test()

    def _gpu_chunk(lst: list[str]) -> list[str]:
        chunk_size = (len(lst) + num_gpus - 1) // num_gpus
        return lst[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]

    train_good = _gpu_chunk(train_good)
    train_bad = _gpu_chunk(train_bad)
    cal_good = _gpu_chunk(cal_good_all)
    cal_bad = _gpu_chunk(cal_bad_all)
    test_good = _gpu_chunk(ood["test_good"])
    test_bad = _gpu_chunk(ood["test_bad"])

    def _extract(texts: list[str], tag: str) -> torch.Tensor:
        print(f"Extracting {tag} (N={len(texts)})...")
        acts = torch.cat(
            [
                extract_activations(b, **common, batch_size=batch_size).cpu()
                for b in _chunk(texts, batch_size)
            ]
        )
        print(f"  {tag}: {tuple(acts.shape)}")
        return acts

    save_dict: dict[str, Any] = {
        "layers": layers,
        "model": model,
        "token_pooling": token_pooling,
    }
    for texts, key in [
        (train_good, "train_good"),
        (train_bad, "train_bad"),
        (cal_good, "cal_good"),
        (cal_bad, "cal_bad"),
        (test_good, "test_good"),
        (test_bad, "test_bad"),
    ]:
        save_dict[f"{key}_acts"] = _extract(texts, key)

    out_file = os.path.join(out_dir, f"acts_{gpu_id}.th")
    torch.save(save_dict, out_file)
    print(f"[GPU {gpu_id}] Saved activations to {out_file}")


# ── Pass 2: probe training + evaluation ──────────────────────────────────────


def aggregate(
    out_dir: str,
    probe_lr: float = 1e-3,
    probe_epochs: int = 100,
    probe_wd: float = 1e-4,
    probe_batch_size: int = 64,
    device: str = "cpu",
) -> dict[str, Any]:
    out_path = Path(out_dir)
    files = sorted(out_path.glob("acts_*.th"))
    assert files, f"No acts_*.th files found in {out_path}"

    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    print(f"Loaded {len(chunks)} GPU activation file(s) from {out_path}")

    _tensor_keys = (
        "train_good_acts",
        "train_bad_acts",
        "cal_good_acts",
        "cal_bad_acts",
        "test_good_acts",
        "test_bad_acts",
    )

    def _cat(key: str) -> torch.Tensor:
        return torch.cat([c[key] for c in chunks], dim=0)

    train_good_acts = _cat("train_good_acts")
    train_bad_acts = _cat("train_bad_acts")
    cal_good_acts = _cat("cal_good_acts")
    cal_bad_acts = _cat("cal_bad_acts")
    test_good_acts = _cat("test_good_acts")
    test_bad_acts = _cat("test_bad_acts")

    cfg = {k: v for k, v in chunks[0].items() if k not in _tensor_keys}
    cfg["layers"] = [int(layer) for layer in cfg["layers"]]
    layers = cfg["layers"]

    n_train_good = len(train_good_acts)
    n_train_bad = len(train_bad_acts)
    n_cal_good = len(cal_good_acts)
    n_cal_bad = len(cal_bad_acts)
    n_test_good = len(test_good_acts)
    n_test_bad = len(test_bad_acts)

    print(f"  train:  good={n_train_good}  bad={n_train_bad}")
    print(f"  cal:    good={n_cal_good}    bad={n_cal_bad}")
    print(f"  test:   good={n_test_good}   bad={n_test_bad}")
    print(f"  layers: {layers}")

    train_labels = torch.cat([torch.ones(n_train_good), torch.zeros(n_train_bad)])
    cal_labels = np.concatenate([np.zeros(n_cal_good), np.ones(n_cal_bad)])
    test_labels = np.concatenate([np.zeros(n_test_good), np.ones(n_test_bad)])

    results: dict[str, Any] = {
        "config": {
            **cfg,
            "probe_lr": probe_lr,
            "probe_epochs": probe_epochs,
            "probe_wd": probe_wd,
            "probe_batch_size": probe_batch_size,
        },
        "n_train_good": n_train_good,
        "n_train_bad": n_train_bad,
        "n_cal_good": n_cal_good,
        "n_cal_bad": n_cal_bad,
        "n_test_good": n_test_good,
        "n_test_bad": n_test_bad,
        "per_layer": {},
        "aggregate": {},
    }

    cal_good_scores = np.zeros((n_cal_good, len(layers)), dtype=np.float32)
    cal_bad_scores = np.zeros((n_cal_bad, len(layers)), dtype=np.float32)
    test_good_scores = np.zeros((n_test_good, len(layers)), dtype=np.float32)
    test_bad_scores = np.zeros((n_test_bad, len(layers)), dtype=np.float32)

    print("\n=================== Training probes ===================")
    layer_auprcs: list[tuple[int, float]] = []

    for li, layer in enumerate(layers):
        train_acts_li = torch.cat(
            [train_good_acts[:, li, :], train_bad_acts[:, li, :]], dim=0
        )
        probe = _train_probe(
            train_acts_li,
            train_labels,
            lr=probe_lr,
            num_epochs=probe_epochs,
            weight_decay=probe_wd,
            batch_size=probe_batch_size,
            device=device,
        )
        cal_good_scores[:, li] = 1.0 - _score_probe(
            probe, cal_good_acts[:, li, :], device, probe_batch_size
        )
        cal_bad_scores[:, li] = 1.0 - _score_probe(
            probe, cal_bad_acts[:, li, :], device, probe_batch_size
        )
        test_good_scores[:, li] = 1.0 - _score_probe(
            probe, test_good_acts[:, li, :], device, probe_batch_size
        )
        test_bad_scores[:, li] = 1.0 - _score_probe(
            probe, test_bad_acts[:, li, :], device, probe_batch_size
        )

        cg, cb = cal_good_scores[:, li], cal_bad_scores[:, li]
        tg, tb = test_good_scores[:, li], test_bad_scores[:, li]
        print(
            f"  Layer {layer:2d}  "
            f"cal_good={cg.mean():.3f}±{cg.std():.3f}  cal_bad={cb.mean():.3f}±{cb.std():.3f}  "
            f"test_good={tg.mean():.3f}±{tg.std():.3f}  test_bad={tb.mean():.3f}±{tb.std():.3f}"
        )

    print("\n=================== Per-layer metrics ===================")
    for li, layer in enumerate(layers):
        cg, cb = cal_good_scores[:, li], cal_bad_scores[:, li]
        tg, tb = test_good_scores[:, li], test_bad_scores[:, li]

        youden_thr = _find_best_f1_threshold(cal_labels, np.concatenate([cg, cb]))
        m = _classification_metrics(
            test_labels,
            np.concatenate([tg, tb]),
            f"layer {layer} probe",
            best_f1_threshold=youden_thr,
        )
        m["good_mean"] = float(tg.mean())
        m["bad_mean"] = float(tb.mean())
        results["per_layer"][f"layer_{layer}"] = m
        layer_auprcs.append((layer, m["auprc"]))

    plot_series: list[tuple[str, NDArray, NDArray]] = []

    def _agg_section(
        label_str: str,
        g_sc: NDArray,
        b_sc: NDArray,
        cal_g_sc: NDArray | None = None,
        cal_b_sc: NDArray | None = None,
    ) -> dict[str, Any]:
        if cal_g_sc is None or cal_b_sc is None:
            cal_g_sc, cal_b_sc = g_sc, b_sc
        cal_lbl = np.concatenate([np.zeros(len(cal_g_sc)), np.ones(len(cal_b_sc))])
        eval_lbl = np.concatenate([np.zeros(len(g_sc)), np.ones(len(b_sc))])
        eval_sc = np.concatenate([g_sc, b_sc])
        ythr = _find_best_f1_threshold(cal_lbl, np.concatenate([cal_g_sc, cal_b_sc]))
        print(
            f"\n  {label_str}  good={g_sc.mean():.4f}±{g_sc.std():.4f}  "
            f"bad={b_sc.mean():.4f}±{b_sc.std():.4f}"
        )
        m = _classification_metrics(
            eval_lbl, eval_sc, label_str, best_f1_threshold=ythr
        )
        m["good_mean"] = float(g_sc.mean())
        m["bad_mean"] = float(b_sc.mean())
        plot_series.append((label_str, eval_lbl, eval_sc))
        return m

    print("\n=================== Aggregate metrics ===================")

    print("\n--- mean across layers ---")
    results["aggregate"]["mean"] = _agg_section(
        "mean probe-score",
        test_good_scores.mean(1),
        test_bad_scores.mean(1),
        cal_g_sc=cal_good_scores.mean(1),
        cal_b_sc=cal_bad_scores.mean(1),
    )

    print("\n--- max across layers (most anomalous layer) ---")
    results["aggregate"]["max"] = _agg_section(
        "max probe-score",
        test_good_scores.max(1),
        test_bad_scores.max(1),
        cal_g_sc=cal_good_scores.max(1),
        cal_b_sc=cal_bad_scores.max(1),
    )

    print("\n--- best single layer (by AUPRC) ---")
    best_layer, best_auprc = max(layer_auprcs, key=lambda x: x[1])
    best_li = layers.index(best_layer)
    print(f"  best layer: {best_layer}  (AUPRC={best_auprc:.4f})")
    m_best = _agg_section(
        f"best-layer probe-score (L{best_layer})",
        test_good_scores[:, best_li],
        test_bad_scores[:, best_li],
        cal_g_sc=cal_good_scores[:, best_li],
        cal_b_sc=cal_bad_scores[:, best_li],
    )
    m_best["best_layer"] = best_layer
    results["aggregate"]["best_layer"] = m_best

    print("\nGenerating plots...")
    try:
        _make_plots(out_path, plot_series)
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}")

    out_json = out_path / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_json}")
    return results


if __name__ == "__main__":
    import fire
    import yaml

    def run(
        config: str = "eval_config.yaml", gpu_id: int = 0, out_dir: str | None = None
    ) -> None:
        import shutil

        with open(config) as f:
            cfg = yaml.safe_load(f)
        resolved_out_dir: str = out_dir or cfg["out_dir"]
        if gpu_id == 0:
            Path(resolved_out_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, Path(resolved_out_dir) / Path(config).name)
        main(
            gpu_id=gpu_id,
            layers=cfg["layers"],
            out_dir=resolved_out_dir,
            model=cfg["model"],
            num_gpus=cfg.get("num_gpus", 4),
            token_pooling=cfg.get("token_pooling", "mean"),
        )

    fire.Fire({"run": run, "aggregate": aggregate})
