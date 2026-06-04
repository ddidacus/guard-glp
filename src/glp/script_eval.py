import io
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import ndarray
from omegaconf import OmegaConf
from PIL import Image
from scipy import linalg

from glp import flow_matching
from glp.denoiser import load_glp

logger = logging.getLogger(__name__)


# =======================
#    Frechet Distance
# =======================
def frechet_distance(
    mu1: ndarray[Any, np.dtype[Any]],
    sigma1: ndarray[Any, np.dtype[Any]],
    mu2: ndarray[Any, np.dtype[Any]],
    sigma2: ndarray[Any, np.dtype[Any]],
    eps: float = 1e-6,
) -> float:
    """
    Calculate the Frechet Distance between two Gaussian distributions.
    Reference: https://github.com/GaParmar/clean-fid/blob/e88c4d6269a4bbf04c04deeb578475b57719acee/cleanfid/fid.py#L37
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError("Training and test mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("Training and test covariances have different dimensions")

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            f"adding {eps} to diagonal of cov estimates"
        )
        logger.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     m = np.max(np.abs(covmean.imag))
        #     raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def rep_fd(
    feats1: ndarray[Any, np.dtype[Any]], feats2: ndarray[Any, np.dtype[Any]]
) -> float:
    """
    Compute the representation Frechet Distance between two sets of features.
    This is the same as fid_from_feats, but not limited to Inception features!
    There's also a faster torch version here: https://docs.pytorch.org/audio/main/_modules/torchaudio/functional/functional.html#frechet_distance,
    but it's less battle tested for numerical stability.
    """
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


# =======================
#          PCA
# =======================
def compute_pca(
    Z: torch.Tensor, k: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    Z_centered = Z - Z.mean(0, keepdim=True)
    _, _, Vt = torch.linalg.svd(Z_centered, full_matrices=False)
    if k is not None:
        Vt = Vt[:k]
    W = Vt.T
    Z_proj = Z_centered @ W
    return W, Z_proj


def plot_pca(
    X: torch.Tensor,
    Y: torch.Tensor,
    label_X: str = "Real",
    label_Y: str = "Generated",
    title: str = "",
    pc_idxs: tuple[int, int] = (0, 1),
    alpha: float = 0.3,
    half_mask: bool = True,
) -> Image.Image:
    Z = torch.cat((X, Y), dim=0)
    W, _ = compute_pca(Z, k=None)

    W_pair = W[:, list(pc_idxs)]
    X2 = ((X - X.mean(0)) @ W_pair).cpu()
    Y2 = ((Y - Y.mean(0)) @ W_pair).cpu()
    fig, ax = plt.subplots()

    def half_mask_fn(n: int, device: torch.device) -> torch.Tensor:
        m = torch.zeros(n, dtype=torch.bool, device=device)
        m[torch.randperm(n, device=device)[: n // 2]] = True
        return m

    m = half_mask_fn(X2.shape[0], X2.device)
    if half_mask:
        ax.scatter(
            X2[~m, 0],
            X2[~m, 1],
            s=8,
            label=label_X,
            color="#f1c232ff",
            zorder=1,
            alpha=alpha,
        )
        ax.scatter(
            Y2[~m, 0],
            Y2[~m, 1],
            s=8,
            label=label_Y,
            color="#e053c3ff",
            zorder=3,
            alpha=alpha,
        )
        ax.scatter(X2[m, 0], X2[m, 1], s=8, color="#f1c232ff", zorder=3, alpha=alpha)
        ax.scatter(Y2[m, 0], Y2[m, 1], s=8, color="#e053c3ff", zorder=1, alpha=alpha)
    else:
        ax.scatter(
            X2[:, 0], X2[:, 1], s=8, label=label_X, color="#f1c232ff", alpha=alpha
        )
        ax.scatter(
            Y2[:, 0], Y2[:, 1], s=8, label=label_Y, color="#e053c3ff", alpha=alpha
        )

    ax.set_xlabel(f"PC {pc_idxs[0] + 1}")
    ax.set_ylabel(f"PC {pc_idxs[1] + 1}")
    ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


def download_ref_acts(ref_folder: str) -> None:
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            "generative-latent-prior/frechet-distance-fineweb-50k",
            "--repo-type",
            "dataset",
            "--local-dir",
            ref_folder,
            "--local-dir-use-symlinks",
            "False",
        ],
        check=True,
    )


@dataclass
class EvalConfig:
    save_folder: str = "runs/eval"
    weights_folder: str | None = "generative-latent-prior/glp-llama8b-d6"
    ckpt_name: str | None = "final"
    ref_folder: str = "data/frechet-distance-fineweb-50k"
    num_timesteps: int = 1000
    batch_size: int | None = None  # set this to a small number if you're getting OOM
    seed: int = 42
    layer_idx: int | None = (
        None  # for glp-llama1b-d12-multi set layer_idx=7, otherwise set to None
    )


def evaluate_sparse_probing(device: str = "cuda:0") -> None:
    default_config = OmegaConf.structured(EvalConfig)
    OmegaConf.set_struct(default_config, False)
    config = OmegaConf.merge(default_config, OmegaConf.from_cli())

    if not os.path.exists(config.ref_folder):
        download_ref_acts(config.ref_folder)
    llm_name = "llama8b" if "llama8b" in config.weights_folder else "llama1b"
    ref_acts = torch.load(f"{config.ref_folder}/{llm_name}.pt")["activations"]
    batch_size = config.batch_size or ref_acts.shape[0]

    model = load_glp(config.weights_folder, device=device, checkpoint=config.ckpt_name)

    generator = torch.Generator().manual_seed(config.seed)
    noise = torch.randn(ref_acts.shape, generator=generator)
    gen_acts = []
    for i in range(0, noise.shape[0], batch_size):
        gen_acts_batch = flow_matching.sample(
            model,
            noise[i : i + batch_size].to(device),
            num_timesteps=config.num_timesteps,
            layer_idx=config.layer_idx,
        )
        gen_acts.append(gen_acts_batch)
    gen_acts = torch.cat(gen_acts, dim=0)
    gen_acts = model.normalizer.denormalize(gen_acts, layer_idx=config.layer_idx)

    gen_acts = gen_acts[:, 0, :].detach().cpu().numpy()
    ref_acts = ref_acts[:, 0, :].detach().cpu().numpy()

    fd = rep_fd(gen_acts, ref_acts)

    weights_name = os.path.basename(config.weights_folder)
    save_file = f"{config.save_folder}/{weights_name}/{config.ckpt_name}.json"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    with open(save_file, "w") as f:
        json.dump({"fd": fd}, f)

    logger.info("FD: %s", fd)


if __name__ == "__main__":
    evaluate_sparse_probing()
