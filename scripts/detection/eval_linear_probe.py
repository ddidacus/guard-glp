import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_classifier import (
    _threshold_metrics,
    _find_best_f1_threshold,
    _classification_metrics,
    _make_plots,
    extract_activations,
    _chunk,
)
from glp.denoiser import load_glp


# ── Classifier model ──────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (N,) raw logits


# ── Probe training / scoring ──────────────────────────────────────────────────

def _train_probe(
    acts: torch.Tensor,     # (N, D) float32
    labels: torch.Tensor,   # (N,) float32  1=good 0=bad
    lr: float,
    num_epochs: int,
    weight_decay: float,
    batch_size: int,
    device: str,
) -> LinearProbe:
    probe   = LinearProbe(acts.shape[1]).to(device)
    opt     = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    ds      = torch.utils.data.TensorDataset(
        acts.to(device=device, dtype=torch.float32),
        labels.to(device=device, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    probe.train()
    for _ in range(num_epochs):
        for x, y in loader:
            opt.zero_grad()
            loss_fn(probe(x), y).backward()
            opt.step()
    return probe


@torch.no_grad()
def _score_probe(probe: LinearProbe, acts: torch.Tensor, device: str, batch_size: int) -> np.ndarray:
    """Sigmoid probability that a sample is benign (in-distribution); higher = more in-dist."""
    probe.eval()
    acts  = acts.to(device=device, dtype=torch.float32)
    probs = []
    for i in range(0, len(acts), batch_size):
        probs.append(torch.sigmoid(probe(acts[i:i + batch_size])).cpu())
    return torch.cat(probs).numpy()


# ── Pass 1: activation extraction (per GPU) ──────────────────────────────────

def main(
    gpu_id: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    num_gpus: int = 4,
    token_pooling: str = "mean",
):
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
    print(f"[+] hf_dataset:    ddidacus/guard-glp-data")
    print(f"[+] gpu:           {gpu_id}/{num_gpus}")
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
    train_dataset       = load_dataset("ddidacus/guard-glp-data", split="train")
    calibration_dataset = load_dataset("ddidacus/guard-glp-data", split="calibration")
    test_dataset        = load_dataset("ddidacus/guard-glp-data", split="test")

    train_good       = [s["prompt"] for s in train_dataset       if not s["adversarial"]]
    train_bad        = [s["prompt"] for s in train_dataset       if s["adversarial"]]
    calibration_good = [s["prompt"] for s in calibration_dataset if not s["adversarial"]]
    calibration_bad  = [s["prompt"] for s in calibration_dataset if s["adversarial"]]
    test_good        = [s["prompt"] for s in test_dataset        if not s["adversarial"]]
    test_bad         = [s["prompt"] for s in test_dataset        if s["adversarial"]]

    def _gpu_chunk(lst: list) -> list:
        chunk_size = (len(lst) + num_gpus - 1) // num_gpus
        return lst[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]

    train_good       = _gpu_chunk(train_good)
    train_bad        = _gpu_chunk(train_bad)
    calibration_good = _gpu_chunk(calibration_good)
    calibration_bad  = _gpu_chunk(calibration_bad)
    test_good        = _gpu_chunk(test_good)
    test_bad         = _gpu_chunk(test_bad)

    def _extract(texts, tag):
        print(f"Extracting {tag} (N={len(texts)})...")
        acts = torch.cat([extract_activations(b, **common, batch_size=batch_size).cpu()
                          for b in _chunk(texts, batch_size)])
        lens = [len(ids) for ids in llm_tokenizer(texts, truncation=True, max_length=2048)["input_ids"]]
        print(f"  {tag}: {tuple(acts.shape)}")
        return acts, lens

    save_dict = {"layers": layers, "model": model,
                 "hf_dataset": "ddidacus/guard-glp-data", "token_pooling": token_pooling}
    for texts, key in [
        (train_good,       "train_good"),
        (train_bad,        "train_bad"),
        (calibration_good, "cal_good"),
        (calibration_bad,  "cal_bad"),
        (test_good,        "test_good"),
        (test_bad,         "test_bad"),
    ]:
        acts, lens = _extract(texts, key)
        save_dict[f"{key}_acts"]          = acts
        save_dict[f"{key}_token_lengths"] = lens

    out_file = os.path.join(out_dir, f"acts_{gpu_id}.th")
    torch.save(save_dict, out_file)
    print(f"[GPU {gpu_id}] Saved activations to {out_file}")


# ── Sanity checks ─────────────────────────────────────────────────────────────

def _run_sanity_checks(
    good_acts: torch.Tensor,
    good_eval_acts: torch.Tensor,
    metric_bad_acts: torch.Tensor,
    bad_acts: torch.Tensor,
    layers: list[int],
    train_labels: torch.Tensor,
    probe_lr: float,
    probe_epochs: int,
    probe_wd: float,
    probe_batch_size: int,
    device: str,
    good_eval_lengths: np.ndarray | None,
    bad_lengths: np.ndarray | None,
    metric_bad_lengths: np.ndarray | None,
    good_eval_scores: np.ndarray | None = None,
    bad_scores: np.ndarray | None = None,
) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    n_good_eval  = len(good_eval_acts)
    n_bad        = len(bad_acts)
    n_metric_bad = len(metric_bad_acts)
    main_labels  = np.concatenate([np.ones(n_good_eval), np.zeros(n_bad)])

    print("\n\n=================== SANITY CHECKS ===================")
    sanity: dict = {}

    # ── norm-only AUROC per layer ─────────────────────────────────────────────
    print("\n[norm-AUROC]  L2 norm of last-token activation as the only feature")
    norm_aurocs = {}
    for li, layer in enumerate(layers):
        g_norm = good_eval_acts[:, li, :].float().norm(dim=1).numpy()
        b_norm = bad_acts[:, li, :].float().norm(dim=1).numpy()
        scores = np.concatenate([g_norm, b_norm])
        auroc  = float(roc_auc_score(main_labels, scores))
        auroc  = max(auroc, 1.0 - auroc)
        print(f"  layer {layer:2d}:  AUROC={auroc:.4f}  "
              f"mean_norm  good={g_norm.mean():.1f}±{g_norm.std():.1f}  "
              f"bad={b_norm.mean():.1f}±{b_norm.std():.1f}")
        norm_aurocs[f"layer_{layer}"] = {"auroc": auroc,
                                          "good_norm_mean": float(g_norm.mean()),
                                          "bad_norm_mean":  float(b_norm.mean())}
    sanity["norm_auroc"] = norm_aurocs

    # ── shuffled-label probe ──────────────────────────────────────────────────
    print("\n[shuffled-label probe]  Probe trained on randomly permuted labels (expected AUROC≈0.5)")
    N_SHUFFLES = 3
    shuffled_aurocs = []
    rng = np.random.default_rng(0)
    for run_i in range(N_SHUFFLES):
        shuffled = train_labels[rng.permutation(len(train_labels))]
        all_layer_aurocs = []
        for li, layer in enumerate(layers):
            train_acts_li = torch.cat([good_acts[:, li, :], metric_bad_acts[:, li, :]], dim=0)
            probe = _train_probe(train_acts_li, shuffled,
                                 lr=probe_lr, num_epochs=probe_epochs,
                                 weight_decay=probe_wd, batch_size=probe_batch_size,
                                 device=device)
            g_sc = _score_probe(probe, good_eval_acts[:, li, :], device, probe_batch_size)
            b_sc = _score_probe(probe, bad_acts[:, li, :], device, probe_batch_size)
            all_layer_aurocs.append(roc_auc_score(main_labels, np.concatenate([g_sc, b_sc])))
        mean_auroc = float(np.mean(all_layer_aurocs))
        shuffled_aurocs.append(mean_auroc)
        print(f"  run {run_i+1}: mean AUROC across layers = {mean_auroc:.4f}")
    sanity["shuffled_probe"] = {
        "mean_auroc": float(np.mean(shuffled_aurocs)),
        "std_auroc":  float(np.std(shuffled_aurocs)),
        "runs": shuffled_aurocs,
    }
    print(f"  → mean={np.mean(shuffled_aurocs):.4f}  std={np.std(shuffled_aurocs):.4f}")

    # ── length-based diagnostics ──────────────────────────────────────────────
    if good_eval_lengths is not None and bad_lengths is not None:
        print("\n[length stats]  Token length distributions")
        print(f"  good_eval : mean={good_eval_lengths.mean():.1f}  "
              f"std={good_eval_lengths.std():.1f}  "
              f"min={good_eval_lengths.min()}  max={good_eval_lengths.max()}")
        print(f"  bad       : mean={bad_lengths.mean():.1f}  "
              f"std={bad_lengths.std():.1f}  "
              f"min={bad_lengths.min()}  max={bad_lengths.max()}")

        all_lengths  = np.concatenate([good_eval_lengths, bad_lengths]).reshape(-1, 1).astype(float)
        lr_clf       = LogisticRegression(max_iter=1000)
        lr_clf.fit(all_lengths, main_labels)
        length_auroc = float(roc_auc_score(main_labels, lr_clf.predict_proba(all_lengths)[:, 1]))
        print(f"\n[length-only AUROC]  LogisticRegression on scalar token length: {length_auroc:.4f}")

        combined_lengths = np.concatenate([good_eval_lengths, bad_lengths])
        q33, q66 = np.percentile(combined_lengths, [33, 66])
        buckets = {
            f"short (≤{int(q33)} tok)":           combined_lengths <= q33,
            f"mid ({int(q33)+1}–{int(q66)} tok)": (combined_lengths > q33) & (combined_lengths <= q66),
            f"long (>{int(q66)} tok)":             combined_lengths > q66,
        }
        print("\n[length-stratified AUROC]  Real probe (mean-layer score), within length bucket")
        strat: dict = {}
        combined_scores = (np.concatenate([good_eval_scores.mean(1), bad_scores.mean(1)])
                           if good_eval_scores is not None and bad_scores is not None else None)
        for name, mask in buckets.items():
            n_in      = mask.sum()
            lbl_in    = main_labels[mask]
            n_good_in = int(lbl_in.sum())
            n_bad_in  = int((lbl_in == 0).sum())
            if n_good_in < 2 or n_bad_in < 2:
                print(f"  {name}: skipped (too few samples: good={n_good_in} bad={n_bad_in})")
                continue
            entry = {"n": int(n_in), "n_good": n_good_in, "n_bad": n_bad_in}
            if combined_scores is not None:
                auroc_in   = float(roc_auc_score(lbl_in, combined_scores[mask]))
                entry["auroc"] = auroc_in
                print(f"  {name}: n={n_in}  (good={n_good_in}  bad={n_bad_in})  AUROC={auroc_in:.4f}")
            else:
                print(f"  {name}: n={n_in}  (good={n_good_in}  bad={n_bad_in})  AUROC=n/a")
            strat[name] = entry

        sanity["length"] = {
            "good_eval_mean": float(good_eval_lengths.mean()),
            "bad_mean":       float(bad_lengths.mean()),
            "length_only_auroc": length_auroc,
            "strat_buckets":  strat,
        }
    else:
        print("\n[length checks]  Skipped — re-run extraction to save token lengths.")
        sanity["length"] = None

    print("=====================================================\n")
    return sanity


# ── Pass 2: probe training + evaluation ──────────────────────────────────────

def aggregate(
    out_dir: str,
    probe_lr: float = 1e-3,
    probe_epochs: int = 100,
    probe_wd: float = 1e-4,
    probe_batch_size: int = 64,
    device: str = "cpu",
) -> dict:
    """Combine GPU shards of activations, train and evaluate linear probes."""
    out_dir = Path(out_dir)
    files   = sorted(out_dir.glob("acts_*.th"))
    assert files, f"No acts_*.th files found in {out_dir}"

    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    print(f"Loaded {len(chunks)} GPU activation file(s) from {out_dir}")

    _tensor_keys = (
        "train_good_acts", "train_bad_acts",
        "cal_good_acts",   "cal_bad_acts",
        "test_good_acts",  "test_bad_acts",
    )
    _length_keys = tuple(k.replace("_acts", "_token_lengths") for k in _tensor_keys)

    def _cat(key):
        return torch.cat([c[key] for c in chunks], dim=0)

    train_good_acts = _cat("train_good_acts")
    train_bad_acts  = _cat("train_bad_acts")
    cal_good_acts   = _cat("cal_good_acts")
    cal_bad_acts    = _cat("cal_bad_acts")
    test_good_acts  = _cat("test_good_acts")
    test_bad_acts   = _cat("test_bad_acts")

    def _cat_lengths(key):
        if chunks[0].get(key) is None:
            return None
        return np.array([l for c in chunks for l in c[key]], dtype=np.int32)

    test_good_token_lengths = _cat_lengths("test_good_token_lengths")
    test_bad_token_lengths  = _cat_lengths("test_bad_token_lengths")
    cal_bad_token_lengths   = _cat_lengths("cal_bad_token_lengths")

    cfg = {k: v for k, v in chunks[0].items()
           if k not in _tensor_keys and k not in _length_keys}
    cfg["layers"] = [int(l) for l in cfg["layers"]]
    layers = cfg["layers"]

    n_train_good = len(train_good_acts)
    n_train_bad  = len(train_bad_acts)
    n_cal_good   = len(cal_good_acts)
    n_cal_bad    = len(cal_bad_acts)
    n_test_good  = len(test_good_acts)
    n_test_bad   = len(test_bad_acts)

    print(f"  train:  good={n_train_good}  bad={n_train_bad}")
    print(f"  cal:    good={n_cal_good}    bad={n_cal_bad}")
    print(f"  test:   good={n_test_good}   bad={n_test_bad}")
    print(f"  layers: {layers}")
    print(f"  probe_lr={probe_lr}  probe_epochs={probe_epochs}  "
          f"probe_wd={probe_wd}  probe_batch_size={probe_batch_size}  device={device}")

    # training labels: 1=good 0=bad (for the probe's BCE loss)
    train_labels = torch.cat([torch.ones(n_train_good), torch.zeros(n_train_bad)])
    # evaluation labels: positive class = adversarial (bad=1, good=0)
    cal_labels   = np.concatenate([np.zeros(n_cal_good),  np.ones(n_cal_bad)])
    test_labels  = np.concatenate([np.zeros(n_test_good), np.ones(n_test_bad)])

    results: dict = {
        "config": {**cfg, "probe_lr": probe_lr, "probe_epochs": probe_epochs,
                   "probe_wd": probe_wd, "probe_batch_size": probe_batch_size},
        "n_train_good": n_train_good, "n_train_bad": n_train_bad,
        "n_cal_good": n_cal_good,     "n_cal_bad":   n_cal_bad,
        "n_test_good": n_test_good,   "n_test_bad":  n_test_bad,
        "per_layer": {}, "aggregate": {},
    }

    cal_good_scores  = np.zeros((n_cal_good,  len(layers)), dtype=np.float32)
    cal_bad_scores   = np.zeros((n_cal_bad,   len(layers)), dtype=np.float32)
    test_good_scores = np.zeros((n_test_good, len(layers)), dtype=np.float32)
    test_bad_scores  = np.zeros((n_test_bad,  len(layers)), dtype=np.float32)

    print("\n=================== Training probes ===================")
    layer_auprcs = []

    for li, layer in enumerate(layers):
        train_acts_li = torch.cat([train_good_acts[:, li, :], train_bad_acts[:, li, :]], dim=0)
        probe = _train_probe(
            train_acts_li, train_labels,
            lr=probe_lr, num_epochs=probe_epochs,
            weight_decay=probe_wd, batch_size=probe_batch_size,
            device=device,
        )
        # probe outputs P(benign); flip to 1 - P(benign) so higher = more adversarial
        cal_good_scores[:, li]  = 1.0 - _score_probe(probe, cal_good_acts[:, li, :],  device, probe_batch_size)
        cal_bad_scores[:, li]   = 1.0 - _score_probe(probe, cal_bad_acts[:, li, :],   device, probe_batch_size)
        test_good_scores[:, li] = 1.0 - _score_probe(probe, test_good_acts[:, li, :], device, probe_batch_size)
        test_bad_scores[:, li]  = 1.0 - _score_probe(probe, test_bad_acts[:, li, :],  device, probe_batch_size)

        cg, cb = cal_good_scores[:, li],  cal_bad_scores[:, li]
        tg, tb = test_good_scores[:, li], test_bad_scores[:, li]
        print(f"  Layer {layer:2d}  "
              f"cal_good={cg.mean():.3f}±{cg.std():.3f}  cal_bad={cb.mean():.3f}±{cb.std():.3f}  "
              f"test_good={tg.mean():.3f}±{tg.std():.3f}  test_bad={tb.mean():.3f}±{tb.std():.3f}")

    print("\n=================== Per-layer metrics ===================")
    for li, layer in enumerate(layers):
        cg, cb = cal_good_scores[:, li],  cal_bad_scores[:, li]
        tg, tb = test_good_scores[:, li], test_bad_scores[:, li]

        youden_thr = _find_best_f1_threshold(cal_labels, np.concatenate([cg, cb]))
        m = _classification_metrics(
            test_labels, np.concatenate([tg, tb]),
            f"layer {layer} probe", best_f1_threshold=youden_thr,
        )
        m["good_mean"] = float(tg.mean())
        m["bad_mean"]  = float(tb.mean())
        results["per_layer"][f"layer_{layer}"] = m
        layer_auprcs.append((layer, m["auprc"]))

    # ── Aggregate strategies ──────────────────────────────────────────────────
    plot_series: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _agg_section(label_str, g_sc, b_sc, cal_g_sc=None, cal_b_sc=None):
        if cal_g_sc is None:
            cal_g_sc, cal_b_sc = g_sc, b_sc
        # positive class = adversarial (bad=1, good=0)
        cal_lbl  = np.concatenate([np.zeros(len(cal_g_sc)), np.ones(len(cal_b_sc))])
        eval_lbl = np.concatenate([np.zeros(len(g_sc)),     np.ones(len(b_sc))])
        eval_sc  = np.concatenate([g_sc, b_sc])
        ythr     = _find_best_f1_threshold(cal_lbl, np.concatenate([cal_g_sc, cal_b_sc]))
        print(f"\n  {label_str}  good={g_sc.mean():.4f}±{g_sc.std():.4f}  "
              f"bad={b_sc.mean():.4f}±{b_sc.std():.4f}")
        m = _classification_metrics(eval_lbl, eval_sc, label_str, best_f1_threshold=ythr)
        m["good_mean"] = float(g_sc.mean())
        m["bad_mean"]  = float(b_sc.mean())
        plot_series.append((label_str, eval_lbl, eval_sc))
        return m

    print("\n=================== Aggregate metrics ===================")

    print("\n--- mean across layers ---")
    results["aggregate"]["mean"] = _agg_section(
        "mean probe-score",
        test_good_scores.mean(1), test_bad_scores.mean(1),
        cal_g_sc=cal_good_scores.mean(1), cal_b_sc=cal_bad_scores.mean(1),
    )

    print("\n--- max across layers (most anomalous layer) ---")
    results["aggregate"]["max"] = _agg_section(
        "max probe-score",
        test_good_scores.max(1), test_bad_scores.max(1),
        cal_g_sc=cal_good_scores.max(1), cal_b_sc=cal_bad_scores.max(1),
    )

    print("\n--- best single layer (by AUROC) ---")
    best_layer, best_auprc = max(layer_auprcs, key=lambda x: x[1])
    best_li = layers.index(best_layer)
    print(f"  best layer: {best_layer}  (AUPRC={best_auprc:.4f})")
    m_best = _agg_section(
        f"best-layer probe-score (L{best_layer})",
        test_good_scores[:, best_li], test_bad_scores[:, best_li],
        cal_g_sc=cal_good_scores[:, best_li], cal_b_sc=cal_bad_scores[:, best_li],
    )
    m_best["best_layer"] = best_layer
    results["aggregate"]["best_layer"] = m_best

    # sanity checks use train split as the "training" reference
    results["sanity"] = _run_sanity_checks(
        good_acts=train_good_acts,
        good_eval_acts=test_good_acts,
        metric_bad_acts=train_bad_acts,
        bad_acts=test_bad_acts,
        layers=layers,
        train_labels=train_labels,
        probe_lr=probe_lr,
        probe_epochs=probe_epochs,
        probe_wd=probe_wd,
        probe_batch_size=probe_batch_size,
        device=device,
        good_eval_lengths=test_good_token_lengths,
        bad_lengths=test_bad_token_lengths,
        metric_bad_lengths=cal_bad_token_lengths,
        good_eval_scores=test_good_scores,
        bad_scores=test_bad_scores,
    )

    print("\nGenerating plots...")
    try:
        _make_plots(out_dir, plot_series)
    except Exception as e:
        print(f"  Warning: could not generate plots: {e}")

    out_json = out_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_json}")
    return results


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
