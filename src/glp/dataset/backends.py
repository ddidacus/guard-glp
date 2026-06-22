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
        # Mirror the baukit path (glp.utils_acts.iter_activations): many causal LMs
        # (e.g. Llama-3.2) ship without a pad token, but batched extraction pads to
        # the longest sequence, so fall back to the eos token and pin padding_side.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = padding_side
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
        # output. vLLM's V1 engine runs the batch *packed* (varlen, no padding), so
        # each traced layer output is (total_tokens, D) — the concatenation of every
        # sequence's tokens in request order — not (B, S, D). We split that packed
        # tensor back into per-sequence rows and scatter them into a padded
        # (B, L, S, D) tensor whose layout matches ``attention_mask``, so the
        # downstream consumer (pooling + writer) is identical to the baukit path.
        # WARNING (unresolved): the captured tensor below does NOT match the HF /
        # baukit residual stream. vLLM's fused-residual Llama layer returns
        # ``(block_output, residual)``; empirically ``output[0]`` is the closest to
        # HF ``layers[i]`` output (cosine ~0.96) but ~4.8x larger in norm, ``output[1]``
        # is near-orthogonal, and their sum is worse. Capturing the true residual
        # stream from nnsight 0.5.0 + vLLM's packed/CUDA-graph execution is an open
        # problem; until it is resolved, treat vllm_nnsight output as NOT
        # interchangeable with the hf_baukit (oracle) datasets. See task notes.
        saved: dict[int, Any] = {}
        with self.llm.trace(batch, max_tokens=1):
            # Under nnsight 0.5.0's vLLM integration, ``self.llm.model`` is already
            # the inner decoder (``LlamaModel`` with ``.layers``), not the
            # ``*ForCausalLM`` wrapper, so the decoder layers live directly under it.
            base = self.llm.model
            for layer in self.layers:
                module_out = base.layers[layer].output
                node = module_out[0] if isinstance(module_out, tuple) else module_out
                saved[layer] = node.save()
        # each (total_tokens, D); stack the configured layers -> (total_tokens, L, D)
        per_layer = [saved[layer] for layer in self.layers]
        packed = torch.stack(per_layer, dim=1).detach().cpu()

        bsz, seq_len = attention_mask.shape
        lengths = attention_mask.sum(dim=1).tolist()  # per-sequence token counts
        total = packed.shape[0]
        if sum(lengths) != total:
            raise ValueError(
                f"vLLM produced {total} packed tokens but the tokenizer counted "
                f"{sum(lengths)} across the batch — the vLLM and HF tokenizations "
                "disagree (e.g. BOS/special-token handling), so packed activations "
                "cannot be aligned to sequences."
            )
        num_layers, dim = packed.shape[1], packed.shape[2]
        acts = torch.zeros(bsz, num_layers, seq_len, dim, dtype=packed.dtype)
        offset = 0
        for b, length in enumerate(int(n) for n in lengths):
            seq = packed[offset : offset + length]  # (length, L, D)
            offset += length
            positions = attention_mask[b].bool()  # (S,); respects padding_side
            acts[b, :, positions, :] = seq.permute(1, 0, 2)  # (L, length, D)
        return acts  # (B, L, S, D)


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
