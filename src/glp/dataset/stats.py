"""Online (single-pass) normalization statistics.

Implements Chan/Welford parallel accumulation of mean and sum-of-squared-
deviations (M2) per hidden dimension. Statistics are accumulated in ``float64``
so ``bfloat16`` activations do not lose precision, and partial results from
independent GPU shards merge exactly into the single-pass result.
"""

import logging
import re
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


def stacked_normalizer_tensors(
    per_layer: dict[int, RunningStats], n_layers_total: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-layer stats into ``(n_layers_total, D)`` mean/var tables.

    Rows are indexed by **absolute layer id** (matching
    ``Normalizer.get_layer_stat``'s ``stat[layer_idx]`` gather); layers without
    stats are filled with NaN so accidentally training on a layer that was never
    measured poisons the loss loudly instead of silently mis-normalizing.
    """
    if not per_layer:
        raise ValueError("per_layer stats are empty")
    if any(layer < 0 or layer >= n_layers_total for layer in per_layer):
        raise ValueError(
            f"layer ids {sorted(per_layer)} out of range for "
            f"n_layers_total={n_layers_total}"
        )
    dim = next(iter(per_layer.values())).dim
    mean = torch.full((n_layers_total, dim), float("nan"), dtype=torch.float32)
    var = torch.full((n_layers_total, dim), float("nan"), dtype=torch.float32)
    for layer, running in per_layer.items():
        mean[layer] = torch.from_numpy(running.mean).to(torch.float32)
        var[layer] = torch.from_numpy(running.var).to(torch.float32)
    return mean, var


def stack_rep_statistics(
    layer_dirs: list[Path], n_layers_total: int, out_path: Path
) -> None:
    """Stack per-layer-dir ``rep_statistics.pt`` files into one ``(n_layers, D)`` file.

    Each directory must be named ``layer_<idx>`` (the builder's convention); its
    ``(D,)`` mean/var land at row ``<idx>`` of the stacked table (NaN elsewhere).
    Makes already-built static per-layer datasets trainable as one multi-layer run.
    """
    stacked: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    dim: int | None = None
    for layer_dir in layer_dirs:
        match = re.search(r"layer_(\d+)", Path(layer_dir).name)
        if not match:
            raise ValueError(f"directory {layer_dir} is not named layer_<idx>")
        layer = int(match.group(1))
        payload = torch.load(Path(layer_dir) / "rep_statistics.pt", map_location="cpu")
        mean, var = payload["mean"].flatten(), payload["var"].flatten()
        if dim is None:
            dim = mean.shape[0]
        elif mean.shape[0] != dim:
            raise ValueError(f"dim mismatch in {layer_dir}: {mean.shape[0]} vs {dim}")
        stacked[layer] = (mean.to(torch.float32), var.to(torch.float32))
    if not stacked:
        raise ValueError("no layer directories provided")
    if any(layer >= n_layers_total for layer in stacked):
        raise ValueError(
            f"layer ids {sorted(stacked)} out of range for "
            f"n_layers_total={n_layers_total}"
        )
    if dim is None:  # unreachable: stacked is non-empty and every entry sets dim
        raise ValueError("could not determine the stats dimensionality")
    mean_table = torch.full((n_layers_total, dim), float("nan"), dtype=torch.float32)
    var_table = torch.full((n_layers_total, dim), float("nan"), dtype=torch.float32)
    for layer, (mean, var) in stacked.items():
        mean_table[layer] = mean
        var_table[layer] = var
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean_table, "var": var_table}, out_path)
    logger.info(
        "stacked %d layer stats into %s ((%d, %d), NaN for missing layers)",
        len(stacked),
        out_path,
        n_layers_total,
        dim,
    )
