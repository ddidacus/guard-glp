import json
import os
import random
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset
from tqdm import tqdm
from glp import flow_matching
from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from transformers import AutoModelForCausalLM, AutoTokenizer

def _threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """ compute eval metrics given labels, scores, threshold """
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    tpr       = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return dict(threshold=threshold, precision=precision,
                recall=tpr, tpr=tpr, fpr=fpr, fnr=fnr,
                tp=tp, fp=fp, fn=fn, tn=tn)

def _find_best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """ minimize balanced error rate: 0.5*(1-TPR) + 0.5*FPR """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    ber = 0.5 * (1 - tpr) + 0.5 * fpr
    return float(thresholds[ber.argmin()])

def _classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    score_name: str,
    verbose: bool = True,
    target_tprs: tuple = tuple(np.arange(0.60, 1.00, 0.05).round(2)),
    best_f1_threshold: float | None = None,
) -> dict:
    """ final scoring function: compute metrics on fixed youden threshold + some TPR thresholds """

    # AUPRC
    from sklearn.metrics import average_precision_score, precision_recall_curve
    auprc = float(average_precision_score(labels, scores))

    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, scores)
    if best_f1_threshold is None:
        f1 = 2 * precision_arr[:-1] * recall_arr[:-1] / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
        best_f1_threshold = float(thresholds[f1.argmax()])

    # threshold that maximises F1 (from calibration set, or this set if none provided)
    youden = _threshold_metrics(labels, scores, best_f1_threshold)

    # thresholds at fixed TPR (recall from PR curve is monotonically decreasing in threshold)
    tpr_arr = recall_arr[:-1][::-1]
    thresholds_asc_by_tpr = thresholds[::-1]
    tpr_results = {}
    for target in target_tprs:
        idx = np.searchsorted(tpr_arr, target)
        idx = min(idx, len(thresholds_asc_by_tpr) - 1)
        tpr_results[f"tpr{int(target*100)}"] = _threshold_metrics(
            labels, scores, float(thresholds_asc_by_tpr[idx])
        )

    if verbose:
        print(f"\n  [{score_name}]")
        print(f"    AUPRC: {auprc:.4f}")
        print(f"    @bestF1(thr={youden['threshold']:.4f}): "
              f"P={youden['precision']:.3f}  TPR={youden['tpr']:.3f}  "
              f"FPR={youden['fpr']:.3f}  FNR={youden['fnr']:.3f}  "
              f"TP={youden['tp']}  FP={youden['fp']}  FN={youden['fn']}  TN={youden['tn']}")
        for key, m in tpr_results.items():
            print(f"    @{key}  (thr={m['threshold']:.4f}): "
                  f"P={m['precision']:.3f}  TPR={m['tpr']:.3f}  "
                  f"FPR={m['fpr']:.3f}  FNR={m['fnr']:.3f}  "
                  f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")

    return dict(auprc=auprc, youden=youden, **tpr_results)

def _make_plots(out_dir: Path, named_scores: list[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """ generate all plots: AUPRC, TPR/FNR """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(named_scores), 1)))

    # ── PR curve ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for (name, labels, scores), color in zip(named_scores, colors):
        precision_arr, recall_arr, _ = precision_recall_curve(labels, scores)
        auprc = average_precision_score(labels, scores)
        ax.plot(recall_arr, precision_arr, label=f"{name}  (AP={auprc:.3f})", color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    # ── TPR and FNR vs threshold ───────────────────────────────────────────
    fig, axes = plt.subplots(1, len(named_scores), figsize=(5 * len(named_scores), 4),
                             squeeze=False)
    for idx, ((name, labels, scores), color) in enumerate(zip(named_scores, colors)):
        ax = axes[0, idx]
        precision_arr, recall_arr, thresholds = precision_recall_curve(labels, scores)
        # precision_recall_curve adds an extra point at the end; align
        tpr_plot = recall_arr[:-1]
        fnr_plot = 1.0 - tpr_plot
        ax.plot(thresholds, tpr_plot, label="TPR", color="steelblue")
        ax.plot(thresholds, fnr_plot, label="FNR", color="tomato")
        # mark F1-maximizing threshold
        f1 = 2 * precision_arr[:-1] * recall_arr[:-1] / (precision_arr[:-1] + recall_arr[:-1] + 1e-8)
        best_thr = thresholds[f1.argmax()]
        ax.axvline(best_thr, color="gray", ls="--", lw=0.9, label="Best F1")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("TPR and FNR vs Threshold", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "tpr_fnr_threshold.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved: {out_dir / 'pr_curves.png'}  |  {out_dir / 'tpr_fnr_threshold.png'}")

def _chunk(lst: list, size: int) -> list[list]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]

def extract_activations(
    texts: list[str],
    llm_model,
    llm_tokenizer,
    diffusion_model,
    device: str = "cuda:0",
    batch_size: int = 8,
    token_pooling: str = "last",
) -> torch.Tensor:
    """ run the LLM and return cached (N, L, D) activations on `device` in bfloat16 """
    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx=token_pooling,
        batch_size=batch_size,
    )
    return activations.to(device=device, dtype=torch.bfloat16)

def extract_log_probs(
    texts: list[str] | None,
    llm_model,
    llm_tokenizer,
    diffusion_model,
    layers: list,
    num_steps: int,
    num_hutchinson_samples: int,
    device: str = "cuda:0",
    batch_size: int = 8,
    method: str = "hutchinson",
    reference_activations: torch.Tensor | None = None,
    K: int = 5,
    num_sigma_bins: int = 100,
    normalize: bool = True,
    precomputed_activations: torch.Tensor | None = None,
) -> dict:
    """ extract log probs depending on method (path integral / dte non-parametric posterior)"""
    if precomputed_activations is not None:
        activations = precomputed_activations.to(device=device, dtype=torch.bfloat16)
    else:
        activations = extract_activations(
            texts, llm_model, llm_tokenizer, diffusion_model,
            device=device, batch_size=batch_size,
        )

    layer_log_probs = []
    layer_probs = []
    layer_expected_sigma = []  # DTE only; None otherwise
    for li, actual_layer in enumerate(layers):
        layer_acts = activations[:, li : li + 1, :]  # (N, 1, D)
        kwargs = dict(
            method=method,
            layer_idx=actual_layer,
            normalize=normalize,
        )
        if method == "hutchinson":
            kwargs.update(
                num_steps=num_steps,
                num_hutchinson_samples=num_hutchinson_samples,
            )
        elif method == "dte":
            assert reference_activations is not None, \
                "method='dte' requires reference_activations"
            kwargs.update(
                reference_latents=reference_activations[:, li : li + 1, :],
                K=K,
                num_sigma_bins=num_sigma_bins,
            )
        result = flow_matching.log_prob(diffusion_model, layer_acts, **kwargs)
        layer_log_probs.append(result.log_prob)
        layer_probs.append(result.prob)
        if method == "dte":
            layer_expected_sigma.append(result.expected_sigma.cpu())

    log_probs = torch.stack(layer_log_probs, dim=1)  # (N, num_layers)
    probs = torch.stack(layer_probs, dim=1)          # (N, num_layers)
    out = {
        "log_probs": log_probs.cpu(),
        "probs": probs.cpu(),
        "activations": activations.cpu().float(),
    }
    if layer_expected_sigma:
        out["expected_sigma"] = torch.stack(layer_expected_sigma, dim=1).cpu()
    return out


@torch.no_grad()
def extract_reconstruction_errors(
    texts: list[str] | None,
    llm_model,
    llm_tokenizer,
    diffusion_model,
    layers: list[int],
    noise_level: float,
    num_timesteps: int,
    device: str = "cuda:0",
    batch_size: int = 8,
    precomputed_activations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct activations via GLP and return per-layer L2 errors (N, L). Higher = more anomalous."""
    if precomputed_activations is not None:
        activations = precomputed_activations.to(device=device, dtype=torch.bfloat16)
    else:
        activations = save_acts(
            hf_model=llm_model,
            hf_tokenizer=llm_tokenizer,
            text=texts,
            tracedict_config=diffusion_model.tracedict_config,
            token_idx="last",
            batch_size=batch_size,
        ).to(device=device, dtype=torch.bfloat16)  # (N, L, D)

    errors_per_layer = []
    for li, actual_layer in enumerate(layers):
        layer_acts = activations[:, li : li + 1, :]  # (N, 1, D)
        normalized = diffusion_model.normalizer.normalize(layer_acts, layer_idx=actual_layer)
        noise = torch.randn_like(normalized)
        noisy, _, timesteps, _ = flow_matching.fm_prepare(
            diffusion_model.scheduler,
            normalized,
            noise,
            u=torch.ones(normalized.shape[0]) * noise_level,
        )
        reconstructed = flow_matching.sample_on_manifold(
            diffusion_model,
            noisy,
            start_timestep=timesteps[0].item(),
            num_timesteps=num_timesteps,
            layer_idx=actual_layer,
        )
        reconstructed = diffusion_model.normalizer.denormalize(reconstructed, layer_idx=actual_layer)
        err = torch.norm(
            (layer_acts - reconstructed).float().reshape(layer_acts.shape[0], -1),
            p=2, dim=1,
        )  # (N,)
        errors_per_layer.append(err.cpu())

    return torch.stack(errors_per_layer, dim=1)  # (N, L)

def main(
    gpu_id: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    num_samples: int | None = None,
    num_steps: int = 100,
    num_hutchinson_samples: int = 1,
    method: str = "hutchinson",
    reference_num_samples: int = 512,
    glp_sample_steps: int = 100,
    dte_K: int = 5,
    dte_num_sigma_bins: int = 100,
    normalize: bool = True,
    noise_level: float = 0.5,
    rec_num_timesteps: int = 100,
    num_gpus: int = 4,
    batch_size: int | None = None,
):
    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

    # load models

    print("[+] Loading models...")

    if model == "1b":
        _default_batch_size = 64
        llm_model_id = "unsloth/Llama-3.2-1B"
        glp_model_id = "generative-latent-prior/glp-llama1b-d12-multi"
    elif model == "8b":
        _default_batch_size = 64
        llm_model_id = "meta-llama/Llama-3.1-8B"
        glp_model_id = "generative-latent-prior/glp-llama8b-d6"
    else:
        raise NotImplementedError()
    if batch_size is None:
        batch_size = _default_batch_size

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(glp_model_id, device=device, checkpoint="final")
    diffusion_model.tracedict_config.layers = layers

    print("================================================")
    print(f"[+] LLM:            {llm_model_id}")
    print(f"[+] GLP:            {glp_model_id}")
    print(f"[+] batch_size:     {batch_size}")
    print(f"[+] num_samples:    {num_samples}")
    print(f"[+] layers:         {layers}")
    print(f"[+] num_steps:      {num_steps}")
    print(f"[+] hutch_samples:  {num_hutchinson_samples}")
    print(f"[+] method:         {method}")
    if method in ("dte", "dte_glp"):
        print(f"[+] dte_K:          {dte_K}")
        print(f"[+] dte_sigma_bins: {dte_num_sigma_bins}")
        print(f"[+] ref_samples:    {reference_num_samples}")
    if method == "dte_glp":
        print(f"[+] glp_sample_steps: {glp_sample_steps}")
    if method == "reconstruction_error":
        print(f"[+] noise_level:    {noise_level}")
        print(f"[+] rec_timesteps:  {rec_num_timesteps}")
    print(f"[+] out_dir:        {out_dir}")
    print("================================================")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    acts_cache_dir = Path(out_dir) / "activations_cache"
    acts_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_split_acts(split_name: str, texts: list[str]) -> torch.Tensor:
        """Return (N, num_layers, D) CPU activations, loading from cache if available."""
        cache_file = acts_cache_dir / f"{split_name}_{gpu_id}.th"
        if cache_file.exists():
            print(f"[+] Loading cached activations for '{split_name}' from {cache_file}")
            return torch.load(cache_file, map_location="cpu", weights_only=True)
        print(f"[+] Extracting activations for '{split_name}' ({len(texts)} samples)...")
        acts = extract_activations(
            texts, llm_model, llm_tokenizer, diffusion_model,
            device=device, batch_size=batch_size,
        ).cpu()
        torch.save(acts, cache_file)
        print(f"    Saved to {cache_file}")
        return acts

    # load dataset
    train_dataset = load_dataset("ddidacus/guard-glp-data", split="train")
    calibration_dataset = load_dataset("ddidacus/guard-glp-data", split="calibration")
    test_dataset = load_dataset("ddidacus/guard-glp-data", split="test")
 
    # organize splits
    train_good = [s["prompt"] for s in train_dataset if not s["adversarial"]]
    train_bad  = [s["prompt"] for s in train_dataset if s["adversarial"]]
    calibration_good = [s["prompt"] for s in calibration_dataset if not s["adversarial"]]
    calibration_bad  = [s["prompt"] for s in calibration_dataset if s["adversarial"]]
    test_good = [s["prompt"] for s in test_dataset if not s["adversarial"]]
    test_bad  = [s["prompt"] for s in test_dataset if s["adversarial"]]

    def _gpu_chunk(lst: list) -> list:
        chunk_size = (len(lst) + num_gpus - 1) // num_gpus
        return lst[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]

    calibration_good = _gpu_chunk(calibration_good)
    calibration_bad  = _gpu_chunk(calibration_bad)
    test_good        = _gpu_chunk(test_good)
    test_bad         = _gpu_chunk(test_bad)

    if not calibration_good or not calibration_bad or not test_good or not test_bad:
        print(f"GPU {gpu_id}: empty chunk, nothing to do.")
        return

    # DTE reference activations (kNN basis)
    reference_activations = None
    if method == "dte":
        ref_prompts = train_good[:reference_num_samples]
        print(f"[+] Building DTE reference set from train benign ({len(ref_prompts)} samples)")
        reference_activations = _get_split_acts(
            f"dte_ref_{reference_num_samples}", ref_prompts
        ).to(device=device, dtype=torch.bfloat16)
        print(f"    reference_activations: {tuple(reference_activations.shape)}")

    # knn ref samples are generated by GLP (approx. fineweb, more logically aligned for classification here)
    elif method == "dte_glp":
        print("[+] Sampling reference data for DTE-GLP...")
        d_input = diffusion_model.denoiser.model.d_input
        print(f"[+] Building DTE reference set by sampling kNN-DTE "
              f"(N={reference_num_samples}, steps={glp_sample_steps}, d_input={d_input})")
        per_layer_refs = []
        for actual_layer in layers:
            noise = torch.randn(reference_num_samples, 1, d_input,
                                device=device, dtype=torch.bfloat16)
            sampled = flow_matching.sample(
                diffusion_model, noise,
                num_timesteps=glp_sample_steps,
                layer_idx=actual_layer,
            )  # (N_ref, 1, D) — in normalized space
            # denormalize so that dte_posterior's normalize=True brings them
            # back to the same space as real LLM activations (avoids double normalization)
            sampled = diffusion_model.normalizer.denormalize(sampled, layer_idx=actual_layer)
            per_layer_refs.append(sampled.cpu())
        # stack into (N_ref, num_layers, D) — same layout as fineweb reference_activations
        reference_activations = torch.cat(per_layer_refs, dim=1).to(device)
        print(f"    reference_activations: {tuple(reference_activations.shape)}")

    print(f"Calibration benign prompts:    {len(calibration_good)}")
    print(f"Calibration malicious prompts: {len(calibration_bad)}")
    print(f"Test benign prompts:           {len(test_good)}")
    print(f"Test malicious prompts:        {len(test_bad)}")

    # Pre-extract activations for all splits (with caching)
    print("[+] Extracting/loading activations for all splits...")
    cal_good_acts  = _get_split_acts("cal_good",  calibration_good)
    cal_bad_acts   = _get_split_acts("cal_bad",   calibration_bad)
    test_good_acts = _get_split_acts("test_good", test_good)
    test_bad_acts  = _get_split_acts("test_bad",  test_bad)

    # organize kwargs
    is_recon = (method == "reconstruction_error")
    _score_method = "dte" if method == "dte_glp" else method
    common_kwargs = dict(
        layers=layers, num_steps=num_steps,
        num_hutchinson_samples=num_hutchinson_samples,
        device=device, batch_size=batch_size,
        method=_score_method,
        reference_activations=reference_activations,
        K=dte_K,
        num_sigma_bins=dte_num_sigma_bins,
        normalize=normalize,
    )
    _recon_kwargs = dict(
        layers=layers, noise_level=noise_level, num_timesteps=rec_num_timesteps,
        device=device, batch_size=batch_size,
    )

    def _score(acts: torch.Tensor):
        """Compute scores from a batch of pre-extracted activations (batch, L, D)."""
        if is_recon:
            return extract_reconstruction_errors(
                None, llm_model, llm_tokenizer, diffusion_model,
                precomputed_activations=acts, **_recon_kwargs
            )
        return extract_log_probs(
            None, llm_model, llm_tokenizer, diffusion_model,
            precomputed_activations=acts, **common_kwargs
        )

    # calibration benign — paired with cal_bad to choose the Youden threshold
    print(f"======= Computing scores for calibration - benign set (GPU {gpu_id}) =======")
    cal_good_results = []
    for i in tqdm(range(0, len(cal_good_acts), batch_size), desc="cal_good", mininterval=30, ncols=120):
        cal_good_results.append(_score(cal_good_acts[i:i+batch_size]))
    assert cal_good_results, "No good train batches processed"

    # calibration bad — Youden threshold selection
    print(f"======= Computing scores for calibration - malicious set (GPU {gpu_id}) =======")
    cal_bad_results = []
    for i in tqdm(range(0, len(cal_bad_acts), batch_size), desc="cal_bad", mininterval=30, ncols=120):
        cal_bad_results.append(_score(cal_bad_acts[i:i+batch_size]))
    assert cal_bad_results, "No bad calibration batches were processed"

    # test benign — held-out good samples used for final evaluation
    print(f"======= Computing scores for test - benign set (GPU {gpu_id}) =======")
    test_good_results = []
    for i in tqdm(range(0, len(test_good_acts), batch_size), desc="test_good", mininterval=30, ncols=120):
        test_good_results.append(_score(test_good_acts[i:i+batch_size]))
    assert test_good_results, "No good eval batches processed"

    # test bad — final evaluation
    print(f"======= Computing scores for test - malicious set (GPU {gpu_id}) =======")
    test_bad_results = []
    for i in tqdm(range(0, len(test_bad_acts), batch_size), desc="test_bad", mininterval=30, ncols=120):
        test_bad_results.append(_score(test_bad_acts[i:i+batch_size]))
    assert test_bad_results, "No bad test batches were processed"

    # store everything
    out_file = os.path.join(out_dir, f"logprob_results_{gpu_id}.th")
    save_dict = {
        "layers": layers,
        "method": method,
        "num_steps": num_steps,
        "num_hutchinson_samples": num_hutchinson_samples,
        "dte_K": dte_K,
        "dte_num_sigma_bins": dte_num_sigma_bins,
        "reference_num_samples": reference_num_samples if method in ("dte", "dte_glp") else 0,
        "glp_sample_steps": glp_sample_steps if method == "dte_glp" else 0,
        "noise_level": noise_level if is_recon else None,
        "rec_num_timesteps": rec_num_timesteps if is_recon else None,
    }

    # reconstruction error classifier
    if is_recon:
        save_dict.update(
            good_recon_errors       = torch.cat(cal_good_results,       dim=0),
            metric_bad_recon_errors = torch.cat(cal_bad_results, dim=0),
            good_eval_recon_errors  = torch.cat(test_good_results,  dim=0),
            bad_recon_errors        = torch.cat(test_bad_results,        dim=0),
        )
    # any other classifier 
    else:
        # calibration
        cal_good_logp       = torch.cat([r["log_probs"] for r in cal_good_results],       dim=0)
        cal_good_probs           = torch.cat([r["probs"]     for r in cal_good_results],       dim=0)
        cal_bad_logp = torch.cat([r["log_probs"] for r in cal_bad_results], dim=0)
        cal_bad_probs     = torch.cat([r["probs"]     for r in cal_bad_results], dim=0)
        # test
        test_good_logp  = torch.cat([r["log_probs"] for r in test_good_results],  dim=0)
        test_bad_logp        = torch.cat([r["log_probs"] for r in test_bad_results],        dim=0)
        test_good_probs      = torch.cat([r["probs"]     for r in test_good_results],  dim=0)
        test_bad_probs            = torch.cat([r["probs"]     for r in test_bad_results],        dim=0)
        # store
        save_dict.update(
            cal_good_logp=cal_good_logp,
            test_good_logp=test_good_logp,
            test_bad_logp=test_bad_logp,
            cal_good_probs=cal_good_probs,
            test_good_probs=test_good_probs,
            test_bad_probs=test_bad_probs,
            cal_bad_logp=cal_bad_logp,
            cal_bad_probs=cal_bad_probs,
        )
        if method in ("dte", "dte_glp"):
            save_dict.update(
                good_expected_sigma       = torch.cat([r["expected_sigma"] for r in cal_good_results],       dim=0),
                good_eval_expected_sigma  = torch.cat([r["expected_sigma"] for r in test_good_results],  dim=0),
                bad_expected_sigma        = torch.cat([r["expected_sigma"] for r in test_bad_results],        dim=0),
                metric_bad_expected_sigma = torch.cat([r["expected_sigma"] for r in cal_bad_results], dim=0),
            )

    torch.save(save_dict, out_file)
    print(f"Saved to {out_file}")


def aggregate(out_dir: str) -> dict:
    """ Load all per-GPU result files in out_dir, concatenate, compute metrics, save JSON.

    Expects files named logprob_results_*.th produced by main().
    Writes results.json to out_dir and returns the metrics dict.

    For DTE, scores both p_clean and expected_sigma (distinct signals).
    Reports metrics for several layer-aggregation strategies:
      mean  — average score across layers (default)
      min   — most anomalous layer (reduces FNR, higher FPR)
      best  — single best layer by per-layer AUPRC
    """
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("logprob_results_*.th"))
    assert files, f"No logprob_results_*.th files found in {out_dir}"

    chunks = [torch.load(f, map_location="cpu", weights_only=False) for f in files]
    print(f"Loaded {len(chunks)} GPU result file(s) from {out_dir}")

    tensor_keys = (
        "cal_good_logp", "test_good_logp", "test_bad_logp",
        "cal_good_probs", "test_good_probs", "test_bad_probs",
        "good_expected_sigma", "good_eval_expected_sigma", "bad_expected_sigma",
        "cal_bad_logp", "cal_bad_probs", "metric_bad_expected_sigma",
        "good_recon_errors", "good_eval_recon_errors", "bad_recon_errors", "metric_bad_recon_errors",
    )
    cat = {k: torch.cat([c[k] for c in chunks], dim=0)
           for k in tensor_keys if chunks[0].get(k) is not None}

    cal_good_logp  = cat.get("cal_good_logp")
    cal_bad_logp   = cat.get("cal_bad_logp")
    test_good_logp = cat.get("test_good_logp")
    test_bad_logp  = cat.get("test_bad_logp")
    # sigmas
    good_es             = cat.get("good_expected_sigma")
    good_eval_es        = cat.get("good_eval_expected_sigma", good_es)
    bad_es              = cat.get("bad_expected_sigma")
    metric_bad_es        = cat.get("metric_bad_expected_sigma")

    cfg = {k: v for k, v in chunks[0].items() if k not in tensor_keys}
    cfg["layers"] = [int(l) for l in cfg["layers"]]
    layers = cfg["layers"]
    method = cfg.get("method", "hutchinson")
    has_dte = method in ("dte", "dte_glp")
    is_recon = (method == "reconstruction_error")

    good_recon_errors       = cat.get("good_recon_errors")
    good_eval_recon_errors  = cat.get("good_eval_recon_errors")
    bad_recon_errors        = cat.get("bad_recon_errors")
    metric_bad_recon_errors = cat.get("metric_bad_recon_errors")

    n_good_cal   = len(good_recon_errors       if is_recon else cal_good_logp)
    n_good_eval  = len(good_eval_recon_errors  if is_recon else test_good_logp)
    n_bad        = len(bad_recon_errors        if is_recon else test_bad_logp)
    n_metric_bad = len(metric_bad_recon_errors if is_recon else cal_bad_logp)
    print(f"  good_cal: {n_good_cal}  |  good_eval: {n_good_eval}  |  bad: {n_bad}  |  "
          f"metric_bad: {n_metric_bad}  |  method: {method}  |  layers: {layers}")

    # positive class = adversarial (bad=1, good=0)
    main_labels = np.concatenate([np.zeros(n_good_eval), np.ones(n_bad)])
    cal_labels  = np.concatenate([np.zeros(n_good_cal),  np.ones(n_metric_bad)])
    results: dict = {"config": cfg, "n_good_cal": n_good_cal, "n_good_eval": n_good_eval,
                     "n_bad": n_bad, "n_metric_bad": n_metric_bad,
                     "per_layer": {}, "aggregate": {}}

    # agrgegation helper — evaluates on the main (eval) set but picks Youden from calibration
    plot_series: list[tuple[str, np.ndarray, np.ndarray]] = []

    def _agg_section(label_str, g_scores, b_scores,
                     cal_g_scores=None, cal_b_scores=None, add_to_plot=True):
        if cal_g_scores is None:
            cal_g_scores, cal_b_scores = g_scores, b_scores
        scores     = np.concatenate([g_scores, b_scores])
        cal_scores = np.concatenate([cal_g_scores, cal_b_scores])
        # positive class = adversarial (bad=1, good=0)
        cal_lbl    = np.concatenate([np.zeros(len(cal_g_scores)), np.ones(len(cal_b_scores))])
        youden_thr = _find_best_f1_threshold(cal_lbl, cal_scores)
        eval_lbl   = np.concatenate([np.zeros(len(g_scores)), np.ones(len(b_scores))])
        print(f"\n  {label_str}  good={g_scores.mean():.4f}±{g_scores.std():.4f}  "
              f"bad={b_scores.mean():.4f}±{b_scores.std():.4f}")
        m = _classification_metrics(eval_lbl, scores, label_str, best_f1_threshold=youden_thr)
        m["good_mean"] = float(g_scores.mean())
        m["bad_mean"]  = float(b_scores.mean())
        if add_to_plot:
            plot_series.append((label_str, eval_lbl, scores))
        return m

    cal_good_scores  = cal_good_logp
    cal_bad_scores   = cal_bad_logp
    test_good_scores = test_good_logp
    test_bad_scores  = test_bad_logp
    main_score_key   = "log_prob"

    # per layer metrics

    print("\n=================== Per-layer metrics ===================")
    layer_auprcs = {}  # score_name -> list of (layer, auprc)

    if is_recon:
        for li, layer in enumerate(layers):
            key = f"layer_{layer}"
            results["per_layer"][key] = {}
            gc_err = good_recon_errors[:, li].numpy()
            g_err  = good_eval_recon_errors[:, li].numpy()
            b_err  = bad_recon_errors[:, li].numpy()
            mb_err = metric_bad_recon_errors[:, li].numpy()
            print(f"\n  Layer {layer}  "
                  f"recon_error good={g_err.mean():.3f}±{g_err.std():.3f}  bad={b_err.mean():.3f}±{b_err.std():.3f}")
            # higher error = more anomalous = adversarial (positive class)
            youden_thr = _find_best_f1_threshold(cal_labels, np.concatenate([gc_err, mb_err]))
            m = _classification_metrics(main_labels, np.concatenate([g_err, b_err]),
                                         f"layer {layer} recon_error", best_f1_threshold=youden_thr)
            m["good_mean"] = float(g_err.mean())
            m["bad_mean"]  = float(b_err.mean())
            results["per_layer"][key]["recon_error"] = m
            layer_auprcs.setdefault("recon_error", []).append((layer, m["auprc"]))
    else:
        for li, layer in enumerate(layers):
            key = f"layer_{layer}"
            results["per_layer"][key] = {}
            gc = cal_good_scores[:, li].numpy()
            mb = cal_bad_scores[:, li].numpy()
            g  = test_good_scores[:, li].numpy()
            b  = test_bad_scores[:, li].numpy()
            print(f"\n  Layer {layer}  "
                  f"{main_score_key} good={g.mean():.4f}±{g.std():.4f}  bad={b.mean():.4f}±{b.std():.4f}")
            # negate log_prob: lower log_prob = more anomalous = adversarial (positive class)
            youden_thr = _find_best_f1_threshold(cal_labels, np.concatenate([-gc, -mb]))
            m = _classification_metrics(main_labels, np.concatenate([-g, -b]),
                                         f"layer {layer} {main_score_key}", best_f1_threshold=youden_thr)
            m["good_mean"] = float(g.mean())
            m["bad_mean"]  = float(b.mean())
            results["per_layer"][key][main_score_key] = m
            layer_auprcs.setdefault(main_score_key, []).append((layer, m["auprc"]))

            if has_dte and good_eval_es is not None:
                gc_es = good_es[:, li].numpy()
                g_es  = good_eval_es[:, li].numpy()
                b_es  = bad_es[:, li].numpy()
                mb_es = metric_bad_es[:, li].numpy()
                print(f"          exp_sigma   good={g_es.mean():.4f}±{g_es.std():.4f}  bad={b_es.mean():.4f}±{b_es.std():.4f}")
                # higher expected_sigma = more anomalous = adversarial (positive class)
                youden_thr_es = _find_best_f1_threshold(cal_labels, np.concatenate([gc_es, mb_es]))
                m_es = _classification_metrics(main_labels, np.concatenate([g_es, b_es]),
                                               f"layer {layer} exp_sigma", best_f1_threshold=youden_thr_es)
                m_es["good_mean"] = float(g_es.mean())
                m_es["bad_mean"]  = float(b_es.mean())
                results["per_layer"][key]["expected_sigma"] = m_es
                layer_auprcs.setdefault("expected_sigma", []).append((layer, m_es["auprc"]))

    print("\n=================== Aggregate metrics ===================")

    if is_recon:
        # higher error = more anomalous = adversarial (positive class)
        print("\n--- mean recon_error across layers ---")
        results["aggregate"]["mean"] = {}
        results["aggregate"]["mean"]["recon_error"] = _agg_section(
            "mean recon_error",
            good_eval_recon_errors.mean(1).numpy(), bad_recon_errors.mean(1).numpy(),
            cal_g_scores=good_recon_errors.mean(1).numpy(),
            cal_b_scores=metric_bad_recon_errors.mean(1).numpy())

        # most anomalous layer wins (highest error)
        print("\n--- max recon_error across layers (most anomalous layer) ---")
        results["aggregate"]["max"] = {}
        results["aggregate"]["max"]["recon_error"] = _agg_section(
            "max recon_error",
            good_eval_recon_errors.max(1).values.numpy(), bad_recon_errors.max(1).values.numpy(),
            cal_g_scores=good_recon_errors.max(1).values.numpy(),
            cal_b_scores=metric_bad_recon_errors.max(1).values.numpy())

        print("\n--- best single layer (by AUPRC) ---")
        results["aggregate"]["best_layer"] = {}
        for score_key, auroc_list in layer_auprcs.items():
            best_layer, best_auprc = max(auroc_list, key=lambda x: x[1])
            best_li = layers.index(best_layer)
            print(f"  best layer for {score_key}: layer {best_layer} (AUPRC={best_auprc:.4f})")
            m = _agg_section(
                f"best-layer {score_key} (L{best_layer})",
                good_eval_recon_errors[:, best_li].numpy(), bad_recon_errors[:, best_li].numpy(),
                cal_g_scores=good_recon_errors[:, best_li].numpy(),
                cal_b_scores=metric_bad_recon_errors[:, best_li].numpy())
            m["best_layer"] = best_layer
            results["aggregate"]["best_layer"][score_key] = m

    else:
        # negate log_prob: lower log_prob = more anomalous = adversarial (positive class)
        print("\n--- mean across layers ---")
        results["aggregate"]["mean"] = {}
        results["aggregate"]["mean"][main_score_key] = _agg_section(
            f"mean {main_score_key}",
            -test_good_scores.mean(1).numpy(), -test_bad_scores.mean(1).numpy(),
            cal_g_scores=-cal_good_scores.mean(1).numpy(), cal_b_scores=-cal_bad_scores.mean(1).numpy())
        if has_dte and good_eval_es is not None:
            # higher expected_sigma = more anomalous = adversarial (positive class)
            results["aggregate"]["mean"]["expected_sigma"] = _agg_section(
                "mean exp_sigma",
                good_eval_es.mean(1).numpy(), bad_es.mean(1).numpy(),
                cal_g_scores=good_es.mean(1).numpy(), cal_b_scores=metric_bad_es.mean(1).numpy())

        # most anomalous layer wins (highest negated log_prob = lowest original log_prob)
        print(f"\n--- max neg {main_score_key} across layers (most anomalous layer) ---")
        results["aggregate"]["max"] = {}
        results["aggregate"]["max"][main_score_key] = _agg_section(
            f"max neg {main_score_key}",
            (-test_good_scores).max(1).values.numpy(), (-test_bad_scores).max(1).values.numpy(),
            cal_g_scores=(-cal_good_scores).max(1).values.numpy(), cal_b_scores=(-cal_bad_scores).max(1).values.numpy())
        if has_dte and good_eval_es is not None:
            # highest expected_sigma across layers
            results["aggregate"]["max"]["expected_sigma"] = _agg_section(
                "max exp_sigma",
                good_eval_es.max(1).values.numpy(), bad_es.max(1).values.numpy(),
                cal_g_scores=good_es.max(1).values.numpy(), cal_b_scores=metric_bad_es.max(1).values.numpy())

        print("\n--- best single layer (by AUPRC) ---")
        results["aggregate"]["best_layer"] = {}
        for score_key, auroc_list in layer_auprcs.items():
            best_layer, best_auprc = max(auroc_list, key=lambda x: x[1])
            best_li = layers.index(best_layer)
            print(f"  best layer for {score_key}: layer {best_layer} (AUPRC={best_auprc:.4f})")
            if score_key == main_score_key:
                # negate log_prob
                g_bl  = -test_good_scores[:, best_li].numpy()
                b_bl  = -test_bad_scores[:, best_li].numpy()
                gc_bl = -cal_good_scores[:, best_li].numpy()
                mb_bl = -cal_bad_scores[:, best_li].numpy()
            else:  # expected_sigma — higher = more adversarial, no negation
                g_bl  = good_eval_es[:, best_li].numpy()
                b_bl  = bad_es[:, best_li].numpy()
                gc_bl = good_es[:, best_li].numpy()
                mb_bl = metric_bad_es[:, best_li].numpy()
            m = _agg_section(f"best-layer {score_key} (L{best_layer})", g_bl, b_bl,
                             cal_g_scores=gc_bl, cal_b_scores=mb_bl)
            m["best_layer"] = best_layer
            results["aggregate"]["best_layer"][score_key] = m

    # generating plots
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
            num_samples=cfg["num_samples"] if "num_samples" in cfg else None,
            num_steps=cfg.get("num_timesteps", 100),
            num_hutchinson_samples=cfg.get("num_hutchinson_samples", 1),
            method=cfg.get("method", "hutchinson"),
            reference_num_samples=cfg.get("reference_num_samples", 512),
            glp_sample_steps=cfg.get("glp_sample_steps", 100),
            dte_K=cfg.get("dte_K", 5),
            dte_num_sigma_bins=cfg.get("dte_num_sigma_bins", 100),
            normalize=cfg.get("normalize", True),
            noise_level=cfg.get("noise_level", 0.5),
            rec_num_timesteps=cfg.get("rec_num_timesteps", 100),
            num_gpus=cfg.get("num_gpus", 4),
            batch_size=cfg.get("batch_size"),
        )

    fire.Fire({"run": run, "aggregate": aggregate})
