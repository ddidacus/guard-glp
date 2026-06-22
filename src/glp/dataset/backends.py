"""Pluggable activation-extraction backends.

A backend turns a list of texts into a stream of per-batch activation tensors
shaped ``(B, L, S, D)`` plus the ``(B, S)`` attention mask (``BatchActs``). The
consumer pools and writes them, so the backend is the only component that knows
how the model is run. Two backends are provided:

* :class:`HFBaukitBackend` — the default. Wraps the in-repo
  :func:`glp.utils_acts.iter_activations` (a refactored ``save_acts``); runs on
  CPU or a single GPU with no optional dependencies, and is the correctness
  oracle for the unit tests.
* :class:`VLLMNNSightBackend` — the high-throughput path used at billion-token
  scale. Requires the optional ``serve`` extra (``vllm``, ``nnsight``); its
  import is guarded so the package works without it.
"""

import importlib
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from glp.utils_acts import iter_activations

if TYPE_CHECKING:
    from glp.dataset.builder import BuildConfig

logger = logging.getLogger(__name__)

# (acts: (B, L, S, D), attention_mask: (B, S)); both detached and on CPU.
BatchActs = tuple[torch.Tensor, torch.Tensor]


class ExtractionBackend(Protocol):
    """Produces activation batches for a list of texts."""

    def iter_batches(self, texts: list[str]) -> Iterator[BatchActs]: ...


class HFBaukitBackend:
    """baukit-based extraction wrapping :func:`glp.utils_acts.iter_activations`."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tracedict_config: dict[str, object],
        batch_size: int,
        max_length: int,
        padding_side: str = "right",
        use_tqdm: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tracedict_config = tracedict_config
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding_side = padding_side
        self.use_tqdm = use_tqdm

    def iter_batches(self, texts: list[str]) -> Iterator[BatchActs]:
        yield from iter_activations(
            self.model,
            self.tokenizer,
            texts,
            self.tracedict_config,
            padding_side=self.padding_side,
            batch_size=self.batch_size,
            max_length=self.max_length,
            use_tqdm=self.use_tqdm,
        )


_SERVE_HINT = (
    "VLLMNNSightBackend requires the optional 'serve' extra "
    "(vllm==0.9.2, nnsight==0.5.0). Install it with `uv sync --extra serve` "
    "(cluster-only; no macOS wheels), or use backend: hf_baukit."
)


class VLLMNNSightBackend:
    """High-throughput extraction via nnsight's vLLM integration.

    NOTE: the exact nnsight<->vLLM tracing API must be validated against
    ``nnsight==0.5.0`` on a GPU node (see the dataset-manager plan's risk note).
    The ``BatchActs`` contract isolates that risk: the consumer is identical to
    the baukit path, which acts as the correctness oracle. This backend is never
    exercised by the CPU test suite.
    """

    def __init__(
        self,
        model_name: str,
        layers: list[int],
        retain: str,
        batch_size: int,
        max_length: int,
        padding_side: str = "right",
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
    ) -> None:
        # Imported dynamically so the package (and pyright under the default,
        # serve-extra-free environment) does not require nnsight to be installed.
        try:
            vllm_module = importlib.import_module("nnsight.modeling.vllm")
        except ImportError as exc:  # pragma: no cover - requires serve extra
            raise ImportError(_SERVE_HINT) from exc

        if retain != "output":
            raise NotImplementedError(
                "VLLMNNSightBackend currently only supports retain='output'"
            )
        self.layers = list(layers)
        self.retain = retain
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding_side = padding_side
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm: Any = vllm_module.VLLM(
            model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_length,
            dtype=dtype,
            task="generate",
            dispatch=True,
        )

    def iter_batches(  # pragma: no cover - requires serve extra + GPU
        self, texts: list[str]
    ) -> Iterator[BatchActs]:
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokenized = self.tokenizer(
                batch,
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            attention_mask = tokenized["attention_mask"]
            yield self._trace_batch(batch, attention_mask), attention_mask

    def _trace_batch(  # pragma: no cover - requires serve extra + GPU
        self, batch: list[str], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Prefill the batch and capture each configured layer's residual-stream
        # output. The tracing API below is the single point to validate against
        # nnsight 0.5.0; everything downstream depends only on the (B, L, S, D)
        # shape produced here.
        saved: dict[int, Any] = {}
        with self.llm.trace(batch, max_tokens=1):
            base = self.llm.model.model
            for layer in self.layers:
                module_out = base.layers[layer].output
                node = module_out[0] if isinstance(module_out, tuple) else module_out
                saved[layer] = node.save()
        per_layer = [saved[layer] for layer in self.layers]  # each (B, S, D)
        acts = torch.stack([t.value for t in per_layer], dim=1)  # (B, L, S, D)
        return acts.detach().cpu()


def make_backend(
    cfg: "BuildConfig", gpu_id: int, device: str | None = None
) -> tuple[ExtractionBackend, PreTrainedTokenizerBase]:
    """Construct the configured backend and its tokenizer.

    ``gpu_id`` is the **data-shard index** (used by ``load_texts`` to stride the
    corpus), decoupled from the CUDA device. ``device`` selects the device: pass
    it explicitly (e.g. ``"cuda:0"`` under a SLURM array task where the one
    allocated GPU is pinned via ``CUDA_VISIBLE_DEVICES``); when ``None`` it falls
    back to ``f"cuda:{gpu_id}"`` if CUDA is available else ``"cpu"`` (the
    single-allocation / local convention where all GPUs are visible).
    """
    if cfg.backend == "hf_baukit":
        if device is None:
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        torch_dtype = (
            torch.bfloat16 if cfg.extract.dtype == "bfloat16" else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, torch_dtype=torch_dtype, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        tracedict_config: dict[str, object] = {
            "layer_prefix": cfg.extract.layer_prefix,
            "layers": list(cfg.extract.layers),
            "retain": cfg.extract.retain,
        }
        backend = HFBaukitBackend(
            model=model,
            tokenizer=tokenizer,
            tracedict_config=tracedict_config,
            batch_size=cfg.extract.batch_size,
            max_length=cfg.extract.max_length,
            padding_side=cfg.extract.padding_side,
        )
        return backend, tokenizer
    if cfg.backend == "vllm_nnsight":
        # vLLM manages its own device placement across the allocated GPUs via
        # tensor parallelism, so the `device` arg does not apply here.
        backend = VLLMNNSightBackend(
            model_name=cfg.model_name,
            layers=list(cfg.extract.layers),
            retain=cfg.extract.retain,
            batch_size=cfg.extract.batch_size,
            max_length=cfg.extract.max_length,
            padding_side=cfg.extract.padding_side,
            dtype="bfloat16" if cfg.extract.dtype == "bfloat16" else "float32",
            tensor_parallel_size=cfg.extract.tensor_parallel_size,
        )
        return backend, backend.tokenizer
    raise ValueError(f"unknown backend: {cfg.backend!r}")
