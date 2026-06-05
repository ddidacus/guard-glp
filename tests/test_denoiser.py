"""CPU-only unit tests for the Normalizer and timestep embedding (no downloads/GPU)."""

import torch

from glp.denoiser import Normalizer, timestep_embedding


def test_normalizer_round_trip() -> None:
    """normalize followed by denormalize recovers the input."""
    mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    var = torch.tensor([1.0, 2.0, 3.0, 4.0])
    norm = Normalizer(mean, var)

    rep = torch.randn(8, 5, 4)
    normalized = norm.normalize(rep)
    recovered = norm.denormalize(normalized)

    assert recovered.shape == rep.shape
    assert torch.allclose(rep, recovered, atol=1e-5)


def test_normalizer_standardizes() -> None:
    """With known stats, normalize subtracts the mean and scales by 1/std."""
    mean = torch.tensor([5.0, -2.0])
    var = torch.tensor([4.0, 9.0])  # std = [2, 3]
    norm = Normalizer(mean, var)

    rep = torch.tensor([[[7.0, 1.0]]])  # (1, 1, 2)
    out = norm.normalize(rep)
    expected = torch.tensor([[[1.0, 1.0]]])  # (7-5)/2=1, (1-(-2))/3=1
    assert torch.allclose(out, expected, atol=1e-6)


def test_timestep_embedding_shape_and_parity() -> None:
    """Sinusoidal embedding has the requested width for even and odd dims."""
    t = torch.tensor([0.0, 1.0, 2.0])
    even = timestep_embedding(t, dim=8)
    odd = timestep_embedding(t, dim=7)
    assert even.shape == (3, 8)
    assert odd.shape == (3, 7)
    assert torch.isfinite(even).all()
