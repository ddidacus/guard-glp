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

from evaluate_classifier import (
    _threshold_metrics,
    _find_best_f1_threshold,
    _classification_metrics,
    _make_plots,
    _chunk,
    extract_activations,
    extract_log_probs,
    extract_reconstruction_errors,
)


def load_ood_test(seed: int = 42):
    """Load 500 wildjailbreak adversarial samples for OOD test (250 benign + 250 harmful)."""
    ds = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
    benign = [s["adversarial"] for s in ds["train"]
              if s["data_type"] == "adversarial_benign" and s["adversarial"] is not None]
    malicious = [s["adversarial"] for s in ds["train"]
                 if s["data_type"] == "adversarial_harmful" and s["adversarial"] is not None]

    rng = np.random.RandomState(seed)
    benign = [benign[i] for i in rng.permutation(len(benign))]
    malicious = [malicious[i] for i in rng.permutation(len(malicious))]

    return dict(test_good=benign[:250], test_bad=malicious[:250])


def main(
    gpu_id: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
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
    print(f"[+] OOD Evaluation (wildjailbreak adversarial)")
    print(f"[+] LLM:            {llm_model_id}")
    print(f"[+] GLP:            {glp_model_id}")
    print(f"[+] batch_size:     {batch_size}")
    print(f"[+] layers:         {layers}")
    print(f"[+] method:         {method}")
    print(f"[+] out_dir:        {out_dir}")
    print("================================================")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    acts_cache_dir = Path(out_dir) / "activations_cache"
    acts_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_split_acts(split_name: str, texts: list[str]) -> torch.Tensor:
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
        return acts

    # calibration from in-distribution data, test from OOD wildjailbreak
    calibration_dataset = load_dataset("ddidacus/guard-glp-data", split="calibration")
    cal_good_all = [s["prompt"] for s in calibration_dataset if not s["adversarial"]]
    cal_bad_all = [s["prompt"] for s in calibration_dataset if s["adversarial"]]
    ood = load_ood_test()

    def _gpu_chunk(lst: list) -> list:
        chunk_size = (len(lst) + num_gpus - 1) // num_gpus
        return lst[gpu_id * chunk_size : (gpu_id + 1) * chunk_size]

    cal_good = _gpu_chunk(cal_good_all)
    cal_bad = _gpu_chunk(cal_bad_all)
    test_good = _gpu_chunk(ood["test_good"])
    test_bad = _gpu_chunk(ood["test_bad"])

    if not cal_good or not cal_bad or not test_good or not test_bad:
        print(f"GPU {gpu_id}: empty chunk, nothing to do.")
        return

    # DTE reference activations from in-distribution train set
    reference_activations = None
    if method == "dte":
        train_dataset = load_dataset("ddidacus/guard-glp-data", split="train")
        ref_prompts = [s["prompt"] for s in train_dataset if not s["adversarial"]][:reference_num_samples]
        print(f"[+] Building DTE reference set from train benign ({len(ref_prompts)} samples)")
        reference_activations = _get_split_acts(
            f"dte_ref_{reference_num_samples}", ref_prompts
        ).to(device=device, dtype=torch.bfloat16)
    elif method == "dte_glp":
        d_input = diffusion_model.denoiser.model.d_input
        per_layer_refs = []
        for actual_layer in layers:
            noise = torch.randn(reference_num_samples, 1, d_input,
                                device=device, dtype=torch.bfloat16)
            sampled = flow_matching.sample(
                diffusion_model, noise, num_timesteps=glp_sample_steps, layer_idx=actual_layer,
            )
            sampled = diffusion_model.normalizer.denormalize(sampled, layer_idx=actual_layer)
            per_layer_refs.append(sampled.cpu())
        reference_activations = torch.cat(per_layer_refs, dim=1).to(device)

    print(f"Calibration benign:    {len(cal_good)}")
    print(f"Calibration malicious: {len(cal_bad)}")
    print(f"Test benign:           {len(test_good)}")
    print(f"Test malicious:        {len(test_bad)}")

    cal_good_acts = _get_split_acts("cal_good", cal_good)
    cal_bad_acts = _get_split_acts("cal_bad", cal_bad)
    test_good_acts = _get_split_acts("test_good", test_good)
    test_bad_acts = _get_split_acts("test_bad", test_bad)

    is_recon = (method == "reconstruction_error")
    _score_method = "dte" if method == "dte_glp" else method
    common_kwargs = dict(
        layers=layers, num_steps=num_steps,
        num_hutchinson_samples=num_hutchinson_samples,
        device=device, batch_size=batch_size,
        method=_score_method,
        reference_activations=reference_activations,
        K=dte_K, num_sigma_bins=dte_num_sigma_bins,
        normalize=normalize,
    )
    _recon_kwargs = dict(
        layers=layers, noise_level=noise_level, num_timesteps=rec_num_timesteps,
        device=device, batch_size=batch_size,
    )

    def _score(acts: torch.Tensor):
        if is_recon:
            return extract_reconstruction_errors(
                None, llm_model, llm_tokenizer, diffusion_model,
                precomputed_activations=acts, **_recon_kwargs
            )
        return extract_log_probs(
            None, llm_model, llm_tokenizer, diffusion_model,
            precomputed_activations=acts, **common_kwargs
        )

    print(f"======= Computing scores (GPU {gpu_id}) =======")
    cal_good_results, cal_bad_results, test_good_results, test_bad_results = [], [], [], []
    for acts, results, desc in [
        (cal_good_acts, cal_good_results, "cal_good"),
        (cal_bad_acts, cal_bad_results, "cal_bad"),
        (test_good_acts, test_good_results, "test_good"),
        (test_bad_acts, test_bad_results, "test_bad"),
    ]:
        for i in tqdm(range(0, len(acts), batch_size), desc=desc, mininterval=30, ncols=120):
            results.append(_score(acts[i:i + batch_size]))

    out_file = os.path.join(out_dir, f"logprob_results_{gpu_id}.th")
    save_dict = {
        "layers": layers, "method": method,
        "num_steps": num_steps, "num_hutchinson_samples": num_hutchinson_samples,
        "dte_K": dte_K, "dte_num_sigma_bins": dte_num_sigma_bins,
        "reference_num_samples": reference_num_samples if method in ("dte", "dte_glp") else 0,
        "noise_level": noise_level if is_recon else None,
        "rec_num_timesteps": rec_num_timesteps if is_recon else None,
    }

    if is_recon:
        save_dict.update(
            good_recon_errors=torch.cat(cal_good_results, dim=0),
            metric_bad_recon_errors=torch.cat(cal_bad_results, dim=0),
            good_eval_recon_errors=torch.cat(test_good_results, dim=0),
            bad_recon_errors=torch.cat(test_bad_results, dim=0),
        )
    else:
        save_dict.update(
            cal_good_logp=torch.cat([r["log_probs"] for r in cal_good_results], dim=0),
            cal_good_probs=torch.cat([r["probs"] for r in cal_good_results], dim=0),
            cal_bad_logp=torch.cat([r["log_probs"] for r in cal_bad_results], dim=0),
            cal_bad_probs=torch.cat([r["probs"] for r in cal_bad_results], dim=0),
            test_good_logp=torch.cat([r["log_probs"] for r in test_good_results], dim=0),
            test_good_probs=torch.cat([r["probs"] for r in test_good_results], dim=0),
            test_bad_logp=torch.cat([r["log_probs"] for r in test_bad_results], dim=0),
            test_bad_probs=torch.cat([r["probs"] for r in test_bad_results], dim=0),
        )
        if method in ("dte", "dte_glp"):
            save_dict.update(
                good_expected_sigma=torch.cat([r["expected_sigma"] for r in cal_good_results], dim=0),
                metric_bad_expected_sigma=torch.cat([r["expected_sigma"] for r in cal_bad_results], dim=0),
                good_eval_expected_sigma=torch.cat([r["expected_sigma"] for r in test_good_results], dim=0),
                bad_expected_sigma=torch.cat([r["expected_sigma"] for r in test_bad_results], dim=0),
            )

    torch.save(save_dict, out_file)
    print(f"Saved to {out_file}")


# aggregate is identical to evaluate_classifier.aggregate
from evaluate_classifier import aggregate


if __name__ == "__main__":
    import yaml
    import fire

    def run(config: str = "eval_config.yaml", gpu_id: int = 0, out_dir: str | None = None):
        import shutil
        with open(config) as f:
            cfg = yaml.safe_load(f)
        out_dir = out_dir or cfg["out_dir"]
        if gpu_id == 0:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, Path(out_dir) / Path(config).name)
        main(
            gpu_id=gpu_id,
            layers=cfg["layers"],
            out_dir=out_dir,
            model=cfg["model"],
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
