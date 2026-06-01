import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import scipy.stats as stats
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from glp import flow_matching
from datasets import load_dataset
from glp.denoiser import load_glp
from glp.utils_acts import save_acts
from glp.script_eval import compute_pca
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Extraction helpers
# ============================================================

@torch.no_grad()
def extract_activations(
    texts: list[str],
    noise_level: float,
    num_timesteps: int,
    llm_model,
    llm_tokenizer,
    diffusion_model,
    layers: list,
    device: str = "cuda:0",
    batch_size: int = 8,
    token_pooling:str = "last") -> dict:

    activations = save_acts(
        hf_model=llm_model,
        hf_tokenizer=llm_tokenizer,
        text=texts,
        tracedict_config=diffusion_model.tracedict_config,
        token_idx=token_pooling,
        batch_size=batch_size,
        use_tqdm=True
    )  # (N, L, D)

    activations = activations.to(device=device, dtype=torch.bfloat16)
    raw_acts = activations.cpu().float()

    reconstructed_layers = []
    noisy_denorm_layers = []

    for li, actual_layer in enumerate(layers):
        layer_acts = activations[:, li:li+1, :]  # (N, 1, D)
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
        reconstructed_layers.append(diffusion_model.normalizer.denormalize(reconstructed, layer_idx=actual_layer))
        noisy_denorm_layers.append(diffusion_model.normalizer.denormalize(noisy, layer_idx=actual_layer))

    reconstructed_acts = torch.cat(reconstructed_layers, dim=1)  # (N, L, D)
    noisy_acts = torch.cat(noisy_denorm_layers, dim=1)            # (N, L, D)
    reconstructed_acts = reconstructed_acts.to(device=activations.device, dtype=activations.dtype)
    noisy_acts = noisy_acts.to(device=activations.device, dtype=activations.dtype)

    delta_orig_recon = torch.abs(activations - reconstructed_acts)

    errors = torch.norm(
        (activations - reconstructed_acts).float().reshape(activations.shape[0], -1),
        p=2,
        dim=1,
    )

    return {
        "activations": raw_acts,
        "noised_activations": noisy_acts,
        "reconstructed_activations": reconstructed_acts,
        "delta_original_reconstructed": delta_orig_recon,
        "reconstruction_errors": errors.cpu(),
    }

def encode_prompts(prompts, batch_size, llm_model, llm_tokenizer, diffusion_model, layers: list,
                   noise_level: float, num_timesteps: int, device="cuda:0", token_pooling:str = "last") -> dict:
    return extract_activations(
        texts=prompts,
        noise_level=noise_level,
        num_timesteps=num_timesteps,
        llm_model=llm_model,
        llm_tokenizer=llm_tokenizer,
        diffusion_model=diffusion_model,
        batch_size=batch_size,
        device=device,
        layers=layers,
        token_pooling=token_pooling
    )

def extract_main(
    gpu_id: int,
    noise_level: float,
    num_timesteps: int,
    layers: list[int],
    out_dir: str,
    model: str = "1b",
    num_gpus: int = 4,
    token_pooling: str = "last"
):
    out_file = os.path.join(out_dir, f"results_{gpu_id}.th")
    if os.path.exists(out_file):
        print(f"[GPU {gpu_id}] shard already exists at {out_file}, skipping")
        return

    torch.manual_seed(42)
    random.seed(42)

    device = f"cuda:{gpu_id}"

    if model == "1b":
        batch_size = 128
        llm_model_id = "unsloth/Llama-3.2-1B"
        glp_model_id = "generative-latent-prior/glp-llama1b-d12-multi"
    elif model == "8b":
        batch_size = 32
        llm_model_id = "meta-llama/Llama-3.1-8B"
        glp_model_id = "generative-latent-prior/glp-llama8b-d6"
    else:
        raise NotImplementedError()

    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    diffusion_model = load_glp(
        glp_model_id, device=device, checkpoint="final"
    )
    diffusion_model.tracedict_config.layers = layers

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = load_dataset("ddidacus/guard-glp-data", split="train")
    good_set = [s["prompt"] for s in train_dataset if not s["adversarial"]]
    bad_set  = [s["prompt"] for s in train_dataset if s["adversarial"]]

    # each GPU processes its contiguous slice of the full dataset
    good_set = good_set[gpu_id::num_gpus]
    bad_set  = bad_set[gpu_id::num_gpus]

    print("================================================")
    print(f"[+] LLM: \t\t{llm_model_id}")
    print(f"[+] GLP: \t\t{glp_model_id}")
    print(f"[+] batch_size: \t{batch_size}")
    print(f"[+] layers: \t\t{diffusion_model.tracedict_config.layers}")
    print(f"[+] noise_level: \t{noise_level}")
    print(f"[+] num_timesteps: \t{num_timesteps}")
    print(f"[+] out_dir: \t\t{out_dir}")
    print(f"[+] shard: \t\t{gpu_id}/{num_gpus}  ({len(good_set)} good, {len(bad_set)} bad)")
    print("================================================")

    print(f"======= Encoding the good set (GPU {gpu_id}) =======")
    result_good = encode_prompts(good_set, batch_size, llm_model, llm_tokenizer, diffusion_model,
                                 layers=layers, noise_level=noise_level, num_timesteps=num_timesteps, device=device, token_pooling=token_pooling)
    activations_good_set    = result_good["activations"]
    reconstructed_good_set  = result_good["reconstructed_activations"]

    print(f"======= Encoding the bad set (GPU {gpu_id}) =======")
    result_bad = encode_prompts(bad_set, batch_size, llm_model, llm_tokenizer, diffusion_model,
                                layers=layers, noise_level=noise_level, num_timesteps=num_timesteps, device=device, token_pooling=token_pooling)
    activations_bad_set    = result_bad["activations"]
    reconstructed_bad_set  = result_bad["reconstructed_activations"]

    torch.save({
        "activations_good_set": activations_good_set,
        "reconstructed_good_set": reconstructed_good_set,
        "activations_bad_set": activations_bad_set,
        "reconstructed_bad_set": reconstructed_bad_set,
    }, out_file)


# ============================================================
# Aggregate / plotting helpers
# ============================================================

def _pca_fit_transform(arrays: list, k: int | None = None):
    """Fit PCA on the concatenation of all arrays, then project each array
    centred by its own mean."""
    tensors = [torch.from_numpy(a.copy()).float() for a in arrays]
    combined = torch.cat(tensors, dim=0)
    W, _ = compute_pca(combined, k=k)   # W: D x k
    return W, [((t - t.mean(0, keepdim=True)) @ W).numpy() for t in tensors]

def _tsne_fit_transform(arrays: list, k: int = 2, perplexity: float = 30.0):
    sizes = [a.shape[0] for a in arrays]
    combined = np.concatenate(arrays, axis=0)
    embedded = TSNE(n_components=k, perplexity=perplexity, random_state=42).fit_transform(combined)
    result, offset = [], 0
    for s in sizes:
        result.append(embedded[offset:offset + s])
        offset += s
    return result

def plot_pca_distributions_layerwise(
    activations_good: torch.Tensor,
    activations_bad: torch.Tensor,
    reconstructed_good: torch.Tensor,
    reconstructed_bad: torch.Tensor,
    layer_indices: list[int],
    out_dir: str,
    prefix: str = "pca_components",
    n_components: int = 10,
    method: str = "pca"):

    method = method.lower()
    assert method in ("pca", "tsne"), f"Unknown method: {method}"
    method_label = "PCA" if method == "pca" else "t-SNE"

    scatter_data = []

    for idx, layer_num in enumerate(layer_indices):
        good      = activations_good[:, idx, :].float().cpu().numpy()
        bad       = activations_bad[:, idx, :].float().cpu().numpy()
        recon_good = reconstructed_good[:, idx, :].float().cpu().numpy()
        recon_bad  = reconstructed_bad[:, idx, :].float().cpu().numpy()

        good_err = np.abs(good - recon_good)
        bad_err  = np.abs(bad  - recon_bad)

        if method == "pca":
            W, (good_proj, bad_proj) = _pca_fit_transform([good, bad], k=n_components)
            recon_good_proj = ((torch.from_numpy(recon_good.copy()).float() -
                                torch.from_numpy(recon_good.copy()).float().mean(0, keepdim=True)) @ W).numpy()
            recon_bad_proj  = ((torch.from_numpy(recon_bad.copy()).float() -
                                torch.from_numpy(recon_bad.copy()).float().mean(0, keepdim=True)) @ W).numpy()
            good_err_t = torch.from_numpy(good_err).float()
            bad_err_t  = torch.from_numpy(bad_err).float()
            good_err_proj = ((good_err_t - good_err_t.mean(0, keepdim=True)) @ W).numpy()
            bad_err_proj  = ((bad_err_t  - bad_err_t.mean(0,  keepdim=True)) @ W).numpy()
        else:
            n_components_tsne = min(n_components, 2)
            all_proj = _tsne_fit_transform(
                [good, bad, recon_good, recon_bad, good_err, bad_err],
                k=n_components_tsne,
            )
            good_proj, bad_proj, recon_good_proj, recon_bad_proj, good_err_proj, bad_err_proj = all_proj

        n_comp_actual = good_proj.shape[1]
        scatter_data.append((layer_num, good_proj, bad_proj, recon_good_proj, recon_bad_proj, good_err_proj, bad_err_proj))

        comp_label = lambda c: f"PC {c+1}" if method == "pca" else f"t-SNE {c+1}"

        for data_pair, fname_suffix, title_prefix in [
            ((good_proj, bad_proj),           f"{prefix}_{layer_num}.png",                f"{method_label} Component Distributions"),
            ((recon_good_proj, recon_bad_proj), f"{prefix}_reconstructions_{layer_num}.png", f"Reconstructed {method_label} Component Distributions"),
            ((good_err_proj, bad_err_proj),    f"{prefix}_errors_{layer_num}.png",          "Per-Component Reconstruction Error"),
        ]:
            gp, bp = data_pair
            fig, axes = plt.subplots(n_comp_actual, 1, figsize=(10, 2.5 * n_comp_actual), sharex=False)
            if n_comp_actual == 1:
                axes = [axes]
            for c in range(n_comp_actual):
                ax = axes[c]
                ax.hist(gp[:, c], bins=50, alpha=0.5, color="tab:blue",   density=True, label="Good")
                ax.hist(bp[:, c], bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad")
                ax.set_ylabel("Density")
                ax.set_title(comp_label(c))
                if c == 0:
                    ax.legend()
            axes[-1].set_xlabel("Value")
            fig.suptitle(f"{title_prefix} — Layer {layer_num}", fontsize=14)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname_suffix), dpi=200)
            plt.close(fig)

    # one figure per layer, three columns
    dim_label  = "PC" if method == "pca" else "t-SNE"
    col_titles  = ["Original activations", "Reconstructed activations", "Reconstruction error"]
    col_xlabel  = f"{dim_label} 1"
    col_ylabel  = f"{dim_label} 2"

    for layer_num, good_proj, bad_proj, recon_good_proj, recon_bad_proj, good_err_proj, bad_err_proj in scatter_data:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for col, (gp, bp) in enumerate([(good_proj, bad_proj), (recon_good_proj, recon_bad_proj), (good_err_proj, bad_err_proj)]):
            ax = axes[col]
            ax.scatter(gp[:, 0], gp[:, 1], s=4, alpha=0.3, color="tab:blue",   label="Benign",   rasterized=True)
            ax.scatter(bp[:, 0], bp[:, 1], s=4, alpha=0.3, color="tab:orange", label="Malicious", rasterized=True)
            ax.set_xlabel(col_xlabel)
            ax.set_ylabel(col_ylabel)
            ax.set_title(col_titles[col])
            ax.legend(markerscale=2)
        fig.tight_layout()
        stem = os.path.join(out_dir, f"{prefix}_scatter2d_layer{layer_num}")
        fig.savefig(stem + ".png", dpi=200, bbox_inches="tight")
        fig.savefig(stem + ".pdf", bbox_inches="tight")
        plt.close(fig)

def plot_error_comparison(
    activations_good: torch.Tensor,
    activations_bad: torch.Tensor,
    reconstructed_good: torch.Tensor,
    reconstructed_bad: torch.Tensor,
    layer_indices: list[int],
    out_dir: str,
    prefix: str):

    fig, axes = plt.subplots(len(layer_indices), 1, figsize=(10, 2.5 * len(layer_indices)), sharex=False)
    if len(layer_indices) == 1:
        axes = [axes]

    good      = activations_good.float().cpu().numpy()
    bad       = activations_bad.float().cpu().numpy()
    recon_good = reconstructed_good.float().cpu().numpy()
    recon_bad  = reconstructed_bad.float().cpu().numpy()

    for idx, layer_num in enumerate(layer_indices):
        good_err_layer = np.linalg.norm(np.abs(good[:, idx, :] - recon_good[:, idx, :]), ord=2, axis=1)
        bad_err_layer  = np.linalg.norm(np.abs(bad[:, idx, :]  - recon_bad[:, idx, :]),  ord=2, axis=1)

        clip = np.percentile(np.concatenate([good_err_layer, bad_err_layer]), 99.9)
        ax = axes[idx]
        ax.hist(good_err_layer, bins=50, alpha=0.5, color="tab:blue",   density=True, label="Good",   range=(0, clip))
        ax.hist(bad_err_layer,  bins=50, alpha=0.5, color="tab:orange", density=True, label="Bad",    range=(0, clip))
        ax.set_xlim(0, clip)
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {layer_num} — Reconstruction Error")
        if idx == 0:
            ax.legend()

    axes[-1].set_xlabel("Value")
    fig.suptitle("Error distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_errors.png"), dpi=200)
    plt.close(fig)

def plot_error_by_layer(error_gap_stats: dict, out_dir: str, prefix: str = "error_by_layer"):
    layer_nums  = [int(k) for k in error_gap_stats]
    good_means  = [error_gap_stats[k]["good_mean"] for k in error_gap_stats]
    good_stds   = [error_gap_stats[k]["good_std"]  for k in error_gap_stats]
    bad_means   = [error_gap_stats[k]["bad_mean"]  for k in error_gap_stats]
    bad_stds    = [error_gap_stats[k]["bad_std"]   for k in error_gap_stats]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.errorbar(layer_nums, good_means, yerr=good_stds, marker="o", capsize=4,
                color="tab:blue",   label="Benign")
    ax.errorbar(layer_nums, bad_means,  yerr=bad_stds,  marker="s", capsize=4,
                color="tab:orange", label="Malicious")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("mean L2 norm")
    ax.set_title("GLP Reconstruction Error")
    ax.set_xticks(layer_nums)
    ax.legend()
    fig.tight_layout()
    stem = os.path.join(out_dir, prefix)
    fig.savefig(stem + ".png", dpi=200)
    fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)

def aggregate_main(results_dir: str, layers: str = "1,7,15", method: str = "pca"):
    if isinstance(layers, str):
        layers = [int(x) for x in layers.split(",")]
    elif isinstance(layers, int):
        layers = [layers]
    else:
        layers = [int(x) for x in layers]

    result_files = sorted(glob.glob(os.path.join(results_dir, "results_*.th")))
    if not result_files:
        print(f"No results_*.th files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(result_files)} result files: {result_files}")

    all_activations_good, all_reconstructed_good = [], []
    all_activations_bad,  all_reconstructed_bad  = [], []

    for path in result_files:
        data = torch.load(path, map_location="cpu", weights_only=True)
        all_activations_good.append(data["activations_good_set"])
        all_reconstructed_good.append(data["reconstructed_good_set"])
        all_activations_bad.append(data["activations_bad_set"])
        all_reconstructed_bad.append(data["reconstructed_bad_set"])

    activations_good  = torch.cat(all_activations_good,  dim=0)
    reconstructed_good = torch.cat(all_reconstructed_good, dim=0)
    activations_bad   = torch.cat(all_activations_bad,   dim=0)
    reconstructed_bad  = torch.cat(all_reconstructed_bad,  dim=0)

    print(f"Aggregated good: {activations_good.shape[0]} samples")
    print(f"Aggregated bad:  {activations_bad.shape[0]} samples")

    plot_error_comparison(
        activations_good=activations_good,
        activations_bad=activations_bad,
        reconstructed_good=reconstructed_good,
        reconstructed_bad=reconstructed_bad,
        layer_indices=layers,
        out_dir=results_dir,
        prefix="aggregated",
    )
    print(f"Error histogram saved to {results_dir}/aggregated_errors.png")

    # error gap stats per layer
    print("\n--- Reconstruction error gap (mean bad - mean good) ---")
    acts_good_np  = activations_good.float().numpy()
    acts_bad_np   = activations_bad.float().numpy()
    recon_good_np = reconstructed_good.float().numpy()
    recon_bad_np  = reconstructed_bad.float().numpy()
    error_gap_stats = {}
    for idx, layer_num in enumerate(layers):
        good_err = np.linalg.norm(np.abs(acts_good_np[:, idx, :] - recon_good_np[:, idx, :]), ord=2, axis=1)
        bad_err  = np.linalg.norm(np.abs(acts_bad_np[:, idx, :]  - recon_bad_np[:, idx, :]),  ord=2, axis=1)
        print(f"  Layer {layer_num:2d}: good={good_err.mean():.4f} ± {good_err.std():.4f}  "
              f"bad={bad_err.mean():.4f} ± {bad_err.std():.4f}  "
              f"gap={bad_err.mean() - good_err.mean():.4f}")
        error_gap_stats[str(layer_num)] = {
            "good_mean": float(good_err.mean()),
            "good_std":  float(good_err.std()),
            "bad_mean":  float(bad_err.mean()),
            "bad_std":   float(bad_err.std()),
            "gap":       float(bad_err.mean() - good_err.mean()),
        }

    gap_json_path = os.path.join(results_dir, "reconstruction_error_gap.json")
    with open(gap_json_path, "w") as f:
        json.dump(error_gap_stats, f, indent=2)
    print(f"\nReconstruction error gap saved to {gap_json_path}")

    plot_error_by_layer(error_gap_stats, results_dir, prefix="error_by_layer")
    print(f"Error-by-layer plot saved to {results_dir}/error_by_layer.png and .pdf")

    plot_pca_distributions_layerwise(
        activations_good=activations_good,
        activations_bad=activations_bad,
        reconstructed_good=reconstructed_good,
        reconstructed_bad=reconstructed_bad,
        layer_indices=layers,
        out_dir=results_dir,
        prefix=method,
        n_components=5,
        method=method,
    )
    print(f"{method.upper()} scatter plots saved to {results_dir}/{method}_scatter2d_layer*.png/.pdf")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import shutil
    import yaml
    import fire

    def run(
        config: str = "eval_config.yaml",
        gpu_id: int = 0,
        aggregate: bool = False,
        results_dir: str = "",
        layers: str = "1,7,15",
        method: str = "pca",
    ):
        if aggregate:
            aggregate_main(results_dir=results_dir, layers=layers, method=method)
            return

        with open(config) as f:
            cfg = yaml.safe_load(f)
        if gpu_id == 0:
            out_dir = cfg["out_dir"]
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(config, Path(out_dir) / Path(config).name)
        extract_main(
            gpu_id=gpu_id,
            noise_level=cfg["noise_level"],
            num_timesteps=cfg["num_timesteps"],
            layers=cfg["layers"],
            out_dir=cfg["out_dir"],
            model=cfg["model"],
            num_gpus=cfg.get("num_gpus", 4),
            token_pooling=cfg.get("token_pooling", "last")
        )

    fire.Fire(run)
