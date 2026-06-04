import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import einops
import numpy as np
import torch
from baukit import TraceDict
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def save_acts(
    hf_model: PreTrainedModel,
    hf_tokenizer: PreTrainedTokenizerBase,
    text: list[str],
    tracedict_config: dict[str, Any],
    padding_side: str = "right",
    token_idx: Literal["last", "mean", "all"] = "last",
    batch_size: int = 10,
    max_length: int = 2048,
    use_tqdm: bool = False,
) -> torch.Tensor:
    with torch.no_grad():
        # set up tracedict
        tracedict_config = dict(tracedict_config)
        retain_attr = tracedict_config.pop("retain")
        if retain_attr not in ["input", "output"]:
            raise ValueError("Must retain exactly one of input or output")
        tracedict_config[f"retain_{retain_attr}"] = True
        if tracedict_config.get("layer_prefix") is not None:
            layer_prefix = tracedict_config.pop("layer_prefix")
            tracedict_config["layers"] = [
                f"{layer_prefix}.{layer}" for layer in tracedict_config["layers"]
            ]
        # set up tokenizer
        if hf_tokenizer.pad_token is None:
            logger.warning("setting tokenizer pad_token to eos_token")
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
        if padding_side != hf_tokenizer.padding_side:
            logger.warning("updating tokenizer padding_side to %s", padding_side)
            hf_tokenizer.padding_side = padding_side
        ret: list[torch.Tensor] = []
        pbar = range(0, len(text), batch_size)
        if use_tqdm:
            pbar = tqdm(pbar)
        for i in pbar:
            start, end = i, min(i + batch_size, len(text))
            minibatch = hf_tokenizer(
                text[start:end],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            )
            minibatch = {k: v.to(hf_model.device) for k, v in minibatch.items()}
            with TraceDict(hf_model, **tracedict_config) as miniret:
                # Run only the base model (skip lm_head) to avoid
                # allocating the huge (batch, seq_len, vocab_size) logits tensor.
                base = getattr(hf_model, "model", hf_model)
                base(**minibatch)
            miniret = [
                getattr(miniret[layer], retain_attr)
                for layer in tracedict_config["layers"]
            ]
            miniret = [x[0] if type(x) is tuple else x for x in miniret]
            miniret = torch.stack(miniret)
            miniret = einops.rearrange(miniret, "l b s d -> b l s d")
            if token_idx == "last":
                last_token_idx = (
                    -1
                    if padding_side == "left"
                    else (minibatch["attention_mask"].sum(dim=1) - 1)
                )
                miniret = (
                    miniret[torch.arange(miniret.shape[0]), :, last_token_idx, :]
                    .detach()
                    .cpu()
                )
            elif token_idx == "mean":
                # miniret: (B, L, S, D); mean-pool over non-padding tokens
                mask = minibatch["attention_mask"].float()  # (B, S)
                lengths = mask.sum(dim=1, keepdim=True)  # (B, 1)
                miniret = (miniret * mask[:, None, :, None]).sum(2) / lengths[
                    :, :, None
                ]  # (B, L, D)
                miniret = miniret.detach().cpu()
            elif token_idx == "all":
                miniret = miniret.detach().cpu()
            else:
                raise NotImplementedError
            ret.append(miniret)
        return torch.cat(ret, dim=0)


@dataclass(kw_only=True)
class MemmapWriter:
    """
    Given a path path/to/dataset/, this will write to:
        path/to/dataset/data_0000.npy
        path/to/dataset/data_0001.npy
        ...
        path/to/dataset/data_indices.npy

    """

    output_dir: Path
    file_size: int  # file size in number of elements
    dtype: np.dtype[Any]

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memmap_files: list[np.memmap[Any, np.dtype[Any]]] = []
        self._new_memmap_file()
        self.cur_idx = 0
        self.indices: list[tuple[int, int, int]] = []  # (file_idx, start_idx, end_idx)

    def _new_memmap_file(self) -> None:
        path = self.output_dir / f"data_{len(self.memmap_files):04d}.npy"
        self.memmap_files.append(
            np.memmap(mode="w+", filename=path, dtype=self.dtype, shape=self.file_size)
        )
        self.cur_idx = 0
        logger.info("Created memmap file %s with size %s", path, self.file_size)

    def write(self, chunk: np.ndarray[Any, np.dtype[Any]]) -> None:
        if chunk.dtype != self.dtype:
            raise ValueError("chunk dtype does not match writer dtype")
        (length,) = chunk.shape
        if length > self.file_size:
            raise ValueError("chunk length exceeds file size")
        if self.cur_idx + length > self.file_size:
            self._new_memmap_file()
        self.memmap_files[-1][self.cur_idx : self.cur_idx + length] = chunk
        self.cur_idx += length
        self.indices.append(
            (len(self.memmap_files) - 1, self.cur_idx - length, self.cur_idx)
        )

    def flush(self) -> None:
        for memmap_file in self.memmap_files:
            memmap_file.flush()
            logger.info("Finished writing to %s", memmap_file.filename)
        indices_path = self.output_dir / "data_indices.npy"
        np.save(indices_path, np.array(self.indices, dtype=np.uint64))
        logger.info("Saved indices to %s", indices_path)


@dataclass()
class MemmapReader:
    data_dir: Path
    dtype: np.dtype[Any]

    def __post_init__(self) -> None:
        indices_path = self.data_dir / "data_indices.npy"
        self.indices = np.load(indices_path)
        logger.info("Loaded %s indices from %s", len(self.indices), indices_path)
        # Dictionary to cache open memmap files
        self._memmap_cache: OrderedDict[int, np.memmap[Any, np.dtype[Any]]] = (
            OrderedDict()
        )

    def __len__(self) -> int:
        return len(self.indices)

    def _get_memmap(self, file_idx: int) -> np.memmap[Any, np.dtype[Any]]:
        """Get or create a memmap for the given file index"""
        if file_idx not in self._memmap_cache:
            filepath = self.data_dir / f"data_{file_idx:04d}.npy"
            self._memmap_cache[file_idx] = np.memmap(
                filename=filepath, mode="r", dtype=self.dtype
            )
            if len(self._memmap_cache) > 3:
                self._memmap_cache.popitem(last=False)
        return self._memmap_cache[file_idx]

    def _get_chunk(self, idx: int) -> np.ndarray[Any, np.dtype[Any]]:
        # Get the file_idx, start_idx, and end_idx for this chunk
        file_idx, start_idx, end_idx = self.indices[idx]
        # Get the memmap for this file
        memmap = self._get_memmap(int(file_idx))
        # Return the chunk
        return memmap[start_idx:end_idx]

    def __getitem__(
        self, idx: int | slice
    ) -> np.ndarray[Any, np.dtype[Any]] | list[np.ndarray[Any, np.dtype[Any]]]:
        """Get the chunk at the given index"""
        if isinstance(idx, slice):
            # Handle slice indexing
            indices = range(*idx.indices(len(self)))
            return [self._get_chunk(i) for i in indices]
        return self._get_chunk(idx)
