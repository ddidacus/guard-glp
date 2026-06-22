"""Activation-dataset consumer ported from the reference ``glp_train.py``.

Ported (and adapted to repo conventions) from
``generative_latent_prior/glp_train.py:55-134``: the in-repo loader that the GLP
trainer consumes. The surrounding training loop (``TrainConfig``, schedulers,
``main``, checkpointing) is intentionally *not* ported here — this module is the
read side of the dataset round-trip only.

A dataset directory is loaded via :class:`~glp.utils_acts.MemmapReader`. Each
sample is stored as a flat ``(D,)`` vector; :class:`ActDataset` returns it shaped
``(1, D)`` so a batch collates to ``(B, 1, D)`` — the ``(batch, seq, dim)`` layout
the single-token GLP expects. Samples stored as ``int16`` are reinterpreted as
``bfloat16`` (the on-disk encoding written by the dataset manager) before being
upcast to ``float32``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from glp.denoiser import Normalizer
from glp.utils_acts import MemmapReader

logger = logging.getLogger(__name__)


class ActDataset(Dataset[dict[str, torch.Tensor]]):
    """A directory of memmapped activations as a torch ``Dataset``."""

    def __init__(self, reader: MemmapReader | list[MemmapReader]) -> None:
        self.reader: list[MemmapReader] = (
            [reader] if isinstance(reader, MemmapReader) else list(reader)
        )

    def __len__(self) -> int:
        return len(self.reader[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}
        # multi-layer convention: directories of the form ``layer_<idx>`` tag
        # each sample with its absolute layer id (used by multi-layer denoisers).
        layer_match = re.search(r"layer_(\d+)", str(self.reader[0].data_dir))
        if layer_match:
            batch["layer_idx"] = torch.tensor(
                int(layer_match.group(1)), dtype=torch.long
            )
        # latents are stored as ``(D,)``; reshape to ``(1, D)`` and (for the
        # multi-reader case) concatenate features along the last dim.
        latents = torch.cat(
            [
                torch.tensor(np.ascontiguousarray(reader[idx]))[None, :]
                for reader in self.reader
            ],
            dim=-1,
        )
        # half-precision activations are stored as int16; reinterpret the bits.
        if latents.dtype == torch.int16:
            latents = latents.view(torch.bfloat16)
        batch["activations"] = latents.float()
        return batch


class ActivationCollator:
    """Stack samples into a batch and apply normalization."""

    def __init__(self, normalizer: Normalizer) -> None:
        self.normalizer = normalizer

    def __call__(self, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            batch: dict[str, torch.Tensor] = {}
            layer_idx: torch.Tensor | None = None
            if "layer_idx" in rows[0]:
                layer_idx = torch.stack([row["layer_idx"] for row in rows], dim=0)
                batch["layer_idx"] = layer_idx
            latents = torch.stack([row["activations"] for row in rows], dim=0)
            batch["latents"] = self.normalizer.normalize(latents, layer_idx=layer_idx)
            return batch


def load_activation_dataset(
    dataset_paths: str | list[str],
) -> ConcatDataset[dict[str, torch.Tensor]]:
    """Load one or more activation directories into a single dataset.

    Each directory must contain ``data_*.npy`` + ``data_indices.npy`` (the memmap)
    and a ``dtype.txt`` naming the on-disk numpy dtype used to read them back.
    """
    paths = [dataset_paths] if isinstance(dataset_paths, str) else list(dataset_paths)
    datasets: list[ActDataset] = []
    for raw_path in paths:
        path = Path(raw_path)
        dtype_path = path / "dtype.txt"
        dtype = np.dtype(dtype_path.read_text().strip().replace("np.", ""))
        reader = MemmapReader(path, dtype)
        datasets.append(ActDataset(reader=reader))
    return ConcatDataset(datasets)


def get_activation_dataloader(
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int,
    normalizer: Normalizer,
    shuffle: bool = True,
) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        collate_fn=ActivationCollator(normalizer),
        num_workers=0,
        pin_memory=False,
    )
