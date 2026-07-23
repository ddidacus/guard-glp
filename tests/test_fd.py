"""CPU-only tests for the FD eval lib (no GPU, no network)."""

from pathlib import Path

import numpy as np
import torch

from glp.denoiser import GLP
from glp.eval.fd import (
    draw_real_pair,
    generate_activations,
    generation_fd,
    rep_fd,
)

# ── FD math ───────────────────────────────────────────────────────────────────


def test_fd_identical_sets_is_zero() -> None:
    x = np.random.default_rng(0).standard_normal((3000, 16))
    assert abs(rep_fd(x, x.copy())) < 1e-6


def test_fd_pure_mean_shift_equals_squared_norm() -> None:
    # shifting every sample by c leaves the covariance unchanged, so FD == ||Δμ||² = dim·c²
    x = np.random.default_rng(1).standard_normal((5000, 8))
    fd = rep_fd(x, x + 2.0)
    assert abs(fd - 8 * 4.0) < 0.5


def test_fd_symmetric() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((2000, 8))
    b = rng.standard_normal((2000, 8)) + 1.0
    assert abs(rep_fd(a, b) - rep_fd(b, a)) < 1e-6


# ── generation + report (tiny GLP, CPU) ───────────────────────────────────────


class _FakeActDataset:
    """Minimal indexable dataset returning {'activations': (1, D)} like ActDataset."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"activations": self.data[i : i + 1]}


def _tiny_glp(dim: int, tmp: Path) -> GLP:
    rep = tmp / "rep_statistics.pt"
    torch.save({"mean": torch.zeros(dim), "var": torch.ones(dim)}, rep)
    return GLP(
        normalizer_config={"rep_statistic": str(rep)},
        denoiser_config={
            "d_input": dim,
            "d_model": 2 * dim,
            "d_mlp": 4 * dim,
            "n_layers": 2,
            "multi_layer_n_layers": None,
        },
    )


def test_generate_activations_shape(tmp_path: Path) -> None:
    dim = 16
    model = _tiny_glp(dim, tmp_path).to("cpu")
    gen = generate_activations(
        model,
        n=40,
        dim=dim,
        num_timesteps=4,
        batch_size=16,
        seed=0,
        layer_idx=None,
        device="cpu",
    )
    assert gen.shape == (40, dim)
    assert np.isfinite(gen).all()


def test_generation_fd_end_to_end(tmp_path: Path) -> None:
    dim = 16
    model = _tiny_glp(dim, tmp_path).to("cpu")
    ds = _FakeActDataset(torch.randn(300, dim))
    real_a, real_b = draw_real_pair(ds, n=60, seed=0)
    assert real_a.shape == (60, dim) and real_b.shape == (60, dim)
    report = generation_fd(
        model,
        real_a,
        real_b,
        num_timesteps=4,
        batch_size=16,
        seed=0,
        layer_idx=None,
        device="cpu",
    )
    assert np.isfinite(report["fd"]) and np.isfinite(report["lower_bound"])
    assert report["n_samples"] == 60 and report["dim"] == dim
