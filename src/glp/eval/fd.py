"""Fréchet Distance (FD) for GLP generative fidelity.

Generation-FD (paper-faithful): fit a Gaussian to a set of activations **generated from
pure noise** by the GLP and to a set of **real** held-out activations, and compute the
Fréchet distance between them, plus the **lower bound** = FD(real_a, real_b) (irreducible
finite-sample error). Lower FD ⇒ the GLP's sampled distribution is closer to the real one.

FD is computed in **raw activation space** (generated samples are denormalized; real
activations are read raw), matching the GLP paper. ``frechet_distance`` / ``rep_fd`` are
the shared FD math (also re-exported by ``glp.script_eval`` for backward compatibility).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from numpy import ndarray
from scipy import linalg
from torch.utils.data import Dataset

from glp import flow_matching
from glp.denoiser import GLP

logger = logging.getLogger(__name__)


# ── Fréchet distance math (relocated from script_eval.py) ────────────────────


def frechet_distance(
    mu1: ndarray[Any, np.dtype[Any]],
    sigma1: ndarray[Any, np.dtype[Any]],
    mu2: ndarray[Any, np.dtype[Any]],
    sigma2: ndarray[Any, np.dtype[Any]],
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between two Gaussians.

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
        logger.warning(
            "fid calculation produces singular product; adding %s to diagonal of cov "
            "estimates",
            eps,
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give a slight imaginary component
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def rep_fd(
    feats1: ndarray[Any, np.dtype[Any]], feats2: ndarray[Any, np.dtype[Any]]
) -> float:
    """Representation Fréchet Distance between two feature sets (mean + full covariance)."""
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)


# ── activation sampling ──────────────────────────────────────────────────────


def _gather_raw(
    dataset: Dataset[dict[str, torch.Tensor]], idx: np.ndarray[Any, np.dtype[Any]]
) -> np.ndarray[Any, np.dtype[Any]]:
    """Stack raw ``(D,)`` activation vectors at the given dataset indices."""
    rows = [dataset[int(i)]["activations"].squeeze(0).float().numpy() for i in idx]
    return np.stack(rows).astype(np.float32)


def draw_real_pair(
    dataset: Dataset[dict[str, torch.Tensor]], n: int, seed: int = 0
) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]]]:
    """Two **disjoint** random samples of ``n`` raw activation vectors.

    The first is compared against generated activations (FD); the pair together gives the
    lower bound FD(real_a, real_b). Indices are sorted so the memmap reads are ~sequential.
    """
    total = len(dataset)  # type: ignore[arg-type]
    n = min(n, total // 2)
    idx = np.random.default_rng(seed).choice(total, size=2 * n, replace=False)
    a = _gather_raw(dataset, np.sort(idx[:n]))
    b = _gather_raw(dataset, np.sort(idx[n:]))
    return a, b


@torch.no_grad()
def generate_activations(
    model: GLP,
    n: int,
    dim: int,
    num_timesteps: int,
    batch_size: int,
    seed: int,
    layer_idx: int | None,
    device: str,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Generate ``n`` activations from pure noise and denormalize to raw ``(n, D)`` space."""
    gen = torch.Generator().manual_seed(seed)
    out: list[np.ndarray[Any, np.dtype[Any]]] = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        noise = torch.randn((b, 1, dim), generator=gen).to(device)
        acts = flow_matching.sample(
            model, noise, num_timesteps=num_timesteps, layer_idx=layer_idx
        )
        acts = model.normalizer.denormalize(acts, layer_idx=layer_idx)
        out.append(acts[:, 0, :].float().detach().cpu().numpy())
        remaining -= b
    return np.concatenate(out, axis=0).astype(np.float32)


def generation_fd(
    model: GLP,
    real_a: np.ndarray[Any, np.dtype[Any]],
    real_b: np.ndarray[Any, np.dtype[Any]],
    num_timesteps: int,
    batch_size: int,
    seed: int,
    layer_idx: int | None,
    device: str,
) -> dict[str, float | int]:
    """Generation-FD report: FD(generated, real_a) and lower bound FD(real_a, real_b)."""
    dim = int(real_a.shape[1])
    generated = generate_activations(
        model, len(real_a), dim, num_timesteps, batch_size, seed, layer_idx, device
    )
    fd = rep_fd(generated, real_a)
    lower_bound = rep_fd(real_a, real_b)
    logger.info(
        "generation FD=%.4f (lower bound=%.4f, n=%d, steps=%d)",
        fd,
        lower_bound,
        len(real_a),
        num_timesteps,
    )
    return {
        "fd": fd,
        "lower_bound": lower_bound,
        "n_samples": int(len(real_a)),
        "num_timesteps": int(num_timesteps),
        "dim": dim,
    }
