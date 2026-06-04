"""CPU-only unit tests for the evaluation metrics (Frechet distance, PCA)."""

import numpy as np
import torch

from glp.script_eval import compute_pca, frechet_distance, rep_fd


def test_frechet_distance_identical_is_zero() -> None:
    """Two identical Gaussians have zero Frechet distance."""
    mu = np.zeros(3)
    sigma = np.eye(3)
    dist = float(frechet_distance(mu, sigma, mu, sigma))
    assert abs(dist) < 1e-6


def test_frechet_distance_mean_shift() -> None:
    """A pure mean shift contributes ||mu1 - mu2||^2 to the distance."""
    sigma = np.eye(2)
    dist = float(frechet_distance(np.zeros(2), sigma, np.array([3.0, 4.0]), sigma))
    assert abs(dist - 25.0) < 1e-4  # 3^2 + 4^2


def test_rep_fd_same_features_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((64, 4))
    assert abs(float(rep_fd(feats, feats))) < 1e-3


def test_compute_pca_shapes() -> None:
    z = torch.randn(20, 5)
    w, z_proj = compute_pca(z, k=2)
    assert w.shape == (5, 2)
    assert z_proj.shape == (20, 2)
