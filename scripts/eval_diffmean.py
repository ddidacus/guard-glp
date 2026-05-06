"""
DiffMean steering-vector classifier on LLM residual-stream activations.

Computes a per-layer steering vector via difference-of-means:
  sv[l] = normalize(mean(calibration_adversarial[l]) - mean(calibration_benign[l]))

Classification score = dot(query_acts, sv) / norm(query_acts) — higher = more adversarial.

Youden threshold derived from calibration; final metrics on test split.

Two-pass workflow:
  Pass 1 — activation extraction (one job per GPU):
      python eval_diffmean.py run --config=cfg.yaml --gpu_id=0
  Pass 2 — steering vector + evaluation (single job, CPU-capable):
      python eval_diffmean.py aggregate --out_dir=results/eval-diffmean
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_classifier import (
    _threshold_metrics,
    _find_youden_threshold,
    _classification_metrics,
    _make_plots,
    extract_activations,
    _chunk,
)
from glp.denoiser import load_glp


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_diffmean(acts: torch.Tensor, sv: np.ndarray) -> np.ndarray:
    """dot(acts, sv) / norm(acts) — higher means more adversarial."""
    sv_t = torch.from_numpy(sv).float()
    a    = acts.float()
    return (a @ sv_t / (a.norm(dim=1) + 1e-8)).numpy()


# ── Pass 1: activation extraction (per GPU) ──────────────────────────────────

def main(
    gpu_id: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    num_gpus: int = 4,
    token_pooling: str = "mean",
):
    """Extract and save activations for calibration and test splits."""
    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

    if model == "1b":
        batch_size   = 64
        llm_model_id = "unsloth/Llama-3.2-1B"
        glp_model_id = "generative-latent-prior/glp-llama1b-d12-multi"
    elif model == "8b":
        batch_size   = 32
        llm_model_id = "meta-llama/Llama-3.1-8B"
        glp_model_id = "generative-latent-prior/glp-llama8b-d6"
    else:
        raise NotImplementedError(f"Unknown model: {model}")

    print("================================================")
    print(f"[+] LLM:           {llm_model_id}")
    print(f"[+] batch_size:    {batch_size}")
    print(f"[+] layers:        {layers}")
    print(f"[+] token_pooling: {token_pooling}")
    print(f"[+] out_dir:       {out_dir}")
    print("================================================")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    llm_model       = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer   = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(glp_model_id, device=device, checkpoint="final")
    diffusion_model.tracedict_config.layers = layers

    common = dict(llm_model=llm_model, llm_tokenizer=llm_tokenizer,
                  diffusion_model=diffusion_model, device=device,
                  token_pooling=token_pooling)

    # load dataset
    calibration_dataset = load_dataset("ddidacus/guard-glp-data", split="calibration")
    test_dataset        = load_dataset("ddidacus/guard-glp-data", split="test")

    calibration_good = [s["prompt"] for s in calibration_dataset if not s["adversarial"]]
    calibration_bad  = [s["prompt"] for s in calibration_dataset if s["adversarial"]]
    test_good        = [s["prompt"] for s in test_dataset if not s["adversarial"]]
    test_bad         = [s["prompt"] for s in test_dataset if s["adversarial"]]

    print(f"Calibration benign:      {len(calibration_good)}")
    print(f"Calibration adversarial: {len(calibration_bad)}")
    print(f"Test benign:             {len(test_good)}")
    print(f"Test adversarial:        {len(test_bad)}")

    def _acts_from_texts(texts, tag):
        from tqdm import tqdm
        print(f"Extracting {tag} (N={len(texts)})...")
        chunks = [extract_activations(b, **common, batch_size=batch_size).cpu()
                  for b in tqdm(_chunk(texts, batch_size), desc=tag, mininterval=30, ncols=120)]
        acts = torch.cat(chunks, dim=0)
        print(f"  {tag}: {tuple(acts.shape)}")
        return acts

    good_acts       = _acts_from_texts(calibration_good, "calibration_benign")
    metric_bad_acts = _acts_from_texts(calibration_bad,  "calibration_adversarial")
    good_eval_acts  = _acts_from_texts(test_good,        "test_benign")
    bad_acts        = _acts_from_texts(test_bad,         "test_adversarial")

    out_file = os.path.join(out_dir, f"acts_{gpu_id}.th")
    torch.save({
        "good_acts":       good_acts,
        "good_eval_acts":  good_eval_acts,
        "metric_bad_acts": metric_bad_acts,
        "bad_acts":        bad_acts,
        "layers":          layers,
        "model":           model,
        "token_pooling":   token_pooling,
    }, out_file)
    print(f"[GPU {gpu_id}] Saved activations to {out_file}")


# ── Pass 2: steering vector computation + evaluation ─────────────────────────

def aggregate(out_dir: str) -> dict:
    """Compute per-layer DiffMean steering vectors and evaluate on the test set.

    Youden threshold is derived from calibration scores and transferred to test,
    matching eval_classifier.py exactly.
    """
    out_dir = Path(out_dir)
    files   = sorted(out_dir.glob("acts_*.th"))
    assert files, f"No acts_*.th files found in {out_dir}"

    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    print(f"Loaded {len(chunks)} GPU activation file(s) from {out_dir}")

    good_acts       = torch.cat([c["good_acts"]       for c in chunks], dim=0)  # calibration benign
    good_eval_acts  = torch.cat([c["good_eval_acts"]  for c in chunks], dim=0)  # test benign
    metric_bad_acts = torch.cat([c["metric_bad_acts"] for c in chunks], dim=0)  # calibration adversarial
    bad_acts        = torch.cat([c["bad_acts"]        for c in chunks], dim=0)  # test adversarial

    cfg = {k: v for k, v in chunks[0].items()
           if k not in ("good_acts", "good_eval_acts", "metric_bad_acts", "bad_acts")}
    cfg["layers"] = [int(l) for l in cfg["layers"]]
    layers = cfg["layers"]

    n_good       = len(good_acts)
    n_good_eval  = len(good_eval_acts)
    n_metric_bad = len(metric_bad_acts)
    n_bad        = len(bad_acts)

    print(f"  cal_benign: {n_good}  |  test_benign: {n_good_eval}  |  "
          f"cal_adv: {n_metric_bad}  |  test_adv: {n_bad}")
    print(f"  layers: {layers}")

    # ── Compute DiffMean steering vectors ─────────────────────────────────────
    print("\n=================== Computing DiffMean steering vectors ===================")
    steering_directions: dict[int, np.ndarray] = {}
    for li, layer in enumerate(layers):
        pos_mean = metric_bad_acts[:, li, :].float().mean(0).numpy()
        neg_mean = good_acts[:, li, :].float().mean(0).numpy()
        d = pos_mean - neg_mean
        norm = np.linalg.norm(d)
        steering_directions[layer] = (d / (norm + 1e-12)).astype(np.float32)
        print(f"  Layer {layer:2d}  sv norm (before norm): {norm:.4f}")

    sv_pt = {layer: torch.from_numpy(sv) for layer, sv in steering_directions.items()}
    torch.save(sv_pt, out_dir / "steering_vector.pt")
    print(f"Steering vector saved to {out_dir / 'steering_vector.pt'}")

    # ── Score all sets ────────────────────────────────────────────────────────
    good_cal_scores   = np.zeros((n_good,       len(layers)), dtype=np.float32)
    good_eval_scores  = np.zeros((n_good_eval,  len(layers)), dtype=np.float32)
    metric_bad_scores = np.zeros((n_metric_bad, len(layers)), dtype=np.float32)
    bad_scores        = np.zeros((n_bad,        len(layers)), dtype=np.float32)

    print("\n=================== Scoring ===================")
    for li, layer in enumerate(layers):
        sv = steering_directions[layer]
        good_cal_scores[:, li]   = _score_diffmean(good_acts[:, li, :],       sv)
        good_eval_scores[:, li]  = _score_diffmean(good_eval_acts[:, li, :],  sv)
        metric_bad_scores[:, li] = _score_diffmean(metric_bad_acts[:, li, :], sv)
        bad_scores[:, li]        = _score_diffmean(bad_acts[:, li, :],        sv)
        gc, ge = good_cal_scores[:, li], good_eval_scores[:, li]
        mb, b  = metric_bad_scores[:, li], bad_scores[:, li]
        print(f"  Layer {layer:2d}  cal_good={gc.mean():.3f}±{gc.std():.3f}  "
              f"cal_bad={mb.mean():.3f}±{mb.std():.3f}  "
              f"test_good={ge.mean():.3f}±{ge.std():.3f}  test_bad={b.mean():.3f}±{b.std():.3f}")

    # labels: 0 = benign, 1 = adversarial
    cal_labels  = np.concatenate([np.zeros(n_good),      np.ones(n_metric_bad)])
    main_labels = np.concatenate([np.zeros(n_good_eval), np.ones(n_bad)])

    results: dict = {
        "config": cfg,
        "n_good_cal": n_good, "n_good_eval": n_good_eval,
        "n_metric_bad": n_metric_bad, "n_bad": n_bad,
        "per_layer": {}, "aggregate": {},
    }

    print("\n=================== Per-layer metrics ===================")
    layer_auprcs = []
    for li, layer in enumerate(layers):
        gc = good_cal_scores[:, li]
        ge = good_eval_scores[:, li]
        mb = metric_bad_scores[:, li]
        b  = bad_scores[:, li]

        youden_thr = _find_youden_threshold(cal_labels, np.concatenate([gc, mb]))
        m = _classification_metrics(
            main_labels, np.concatenate([ge, b]),
            f"layer {layer} diffmean", youden_threshold=youden_thr,
        )
        m["good_mean"] = float(ge.mean())
        m["bad_mean"]  = float(b.mean())
        results["per_layer"][f"layer_{layer}"] = m
        layer_auprcs.append((layer, m["auprc"]))

    # ── Aggregate strategies ──────────────────────────────────────────────────
    plot_series: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _agg_section(label_str, g_sc, b_sc, cal_g_sc=None, cal_b_sc=None):
        if cal_g_sc is None:
            cal_g_sc, cal_b_sc = g_sc, b_sc
        cal_lbl  = np.concatenate([np.zeros(len(cal_g_sc)), np.ones(len(cal_b_sc))])
        eval_lbl = np.concatenate([np.zeros(len(g_sc)),     np.ones(len(b_sc))])
        ythr     = _find_youden_threshold(cal_lbl, np.concatenate([cal_g_sc, cal_b_sc]))
        eval_sc  = np.concatenate([g_sc, b_sc])
        print(f"\n  {label_str}  good={g_sc.mean():.4f}±{g_sc.std():.4f}  "
              f"bad={b_sc.mean():.4f}±{b_sc.std():.4f}")
        m = _classification_metrics(eval_lbl, eval_sc, label_str, youden_threshold=ythr)
        m["good_mean"] = float(g_sc.mean())
        m["bad_mean"]  = float(b_sc.mean())
        plot_series.append((label_str, eval_lbl, eval_sc))
        return m

    print("\n=================== Aggregate metrics ===================")

    print("\n--- mean across layers ---")
    results["aggregate"]["mean"] = _agg_section(
        "mean diffmean-score",
        good_eval_scores.mean(1), bad_scores.mean(1),
        cal_g_sc=good_cal_scores.mean(1), cal_b_sc=metric_bad_scores.mean(1),
    )

    print("\n--- max across layers (most adversarial layer) ---")
    results["aggregate"]["max"] = _agg_section(
        "max diffmean-score",
        good_eval_scores.max(1), bad_scores.max(1),
        cal_g_sc=good_cal_scores.max(1), cal_b_sc=metric_bad_scores.max(1),
    )

    print("\n--- best single layer (by AUROC) ---")
    best_layer, best_auprc = max(layer_auprcs, key=lambda x: x[1])
    best_li = layers.index(best_layer)
    print(f"  best layer: {best_layer}  (AUPRC={best_auprc:.4f})")
    m_best = _agg_section(
        f"best-layer diffmean-score (L{best_layer})",
        good_eval_scores[:, best_li], bad_scores[:, best_li],
        cal_g_sc=good_cal_scores[:, best_li], cal_b_sc=metric_bad_scores[:, best_li],
    )
    m_best["best_layer"] = best_layer
    results["aggregate"]["best_layer"] = m_best

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    try:
        _make_plots(out_dir, plot_series)
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}")

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, torch.Tensor): return o.tolist()
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.integer, np.floating)): return o.item()
            return super().default(o)

    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, cls=_Encoder)
    print(f"Saved to {out_json}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    import fire

    def run(config: str = "eval_config.yaml", gpu_id: int = 0):
        import shutil
        with open(config) as f:
            cfg = yaml.safe_load(f)
        if gpu_id == 0:
            out_dir = cfg["out_dir"]
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, Path(out_dir) / Path(config).name)
        main(
            gpu_id=gpu_id,
            layers=cfg["layers"],
            out_dir=cfg["out_dir"],
            model=cfg["model"],
            num_gpus=cfg.get("num_gpus", 4),
            token_pooling=cfg.get("token_pooling", "mean"),
        )

    fire.Fire({"run": run, "aggregate": aggregate})
