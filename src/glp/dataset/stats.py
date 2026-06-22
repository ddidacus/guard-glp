"""Online (single-pass) normalization statistics.

Implements Chan/Welford parallel accumulation of mean and sum-of-squared-
deviations (M2) per hidden dimension. Statistics are accumulated in ``float64``
so ``bfloat16`` activations do not lose precision, and partial results from
independent GPU shards merge exactly into the single-pass result.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from glp.denoiser import Normalizer

logger = logging.getLogger(__name__)

F64 = npt.NDArray[np.float64]


@dataclass
class RunningStats:
    """Streaming per-dimension mean and variance over ``(N, D)`` samples."""

    dim: int
    count: float
    mean: F64
    m2: F64

    @classmethod
    def zeros(cls, dim: int) -> "RunningStats":
        return cls(
            dim=dim,
            count=0.0,
            mean=np.zeros(dim, dtype=np.float64),
            m2=np.zeros(dim, dtype=np.float64),
        )

    def update(self, x: torch.Tensor) -> None:
        """Fold a batch of ``(N, D)`` samples into the running statistics."""
        arr = x.detach().to(torch.float64).cpu().numpy()
        if arr.ndim != 2:
            raise ValueError(f"expected (N, D) samples, got shape {arr.shape}")
        if arr.shape[1] != self.dim:
            raise ValueError(f"dim mismatch: expected {self.dim}, got {arr.shape[1]}")
        n_b = arr.shape[0]
        if n_b == 0:
            return
        batch_mean = arr.mean(axis=0)
        batch_m2 = ((arr - batch_mean) ** 2).sum(axis=0)
        self._merge_moments(float(n_b), batch_mean, batch_m2)

    def merge(self, other: "RunningStats") -> None:
        """Merge another shard's statistics into this one (exact, order-free)."""
        if self.dim != other.dim:
            raise ValueError(f"dim mismatch: {self.dim} vs {other.dim}")
        if other.count == 0:
            return
        self._merge_moments(other.count, other.mean, other.m2)

    def _merge_moments(self, n_b: float, mean_b: F64, m2_b: F64) -> None:
        if self.count == 0:
            self.count = n_b
            self.mean = mean_b.copy()
            self.m2 = m2_b.copy()
            return
        delta = mean_b - self.mean
        new_count = self.count + n_b
        self.mean = self.mean + delta * (n_b / new_count)
        self.m2 = self.m2 + m2_b + delta**2 * (self.count * n_b / new_count)
        self.count = new_count

    @property
    def var(self) -> F64:
        """Population variance (matches ``Normalizer.check_normalized``)."""
        if self.count == 0:
            raise ValueError("cannot compute variance with zero samples")
        return self.m2 / self.count

    def save_partial(self, path: str | Path) -> None:
        torch.save(
            {
                "dim": self.dim,
                "count": self.count,
                "mean": torch.from_numpy(self.mean),
                "m2": torch.from_numpy(self.m2),
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RunningStats":
        payload = torch.load(Path(path), map_location="cpu")
        return cls(
            dim=int(payload["dim"]),
            count=float(payload["count"]),
            mean=payload["mean"].numpy().astype(np.float64),
            m2=payload["m2"].numpy().astype(np.float64),
        )

    def to_normalizer(self) -> Normalizer:
        """Build a :class:`~glp.denoiser.Normalizer` with ``(D,)`` mean/var."""
        mean = torch.from_numpy(self.mean).to(torch.float32)
        var = torch.from_numpy(self.var).to(torch.float32)
        return Normalizer(mean, var)
