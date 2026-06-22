# Dataset Manager — Design & Implementation Plan

## Context

GuardGLP reframes adversarial-prompt detection as anomaly detection: train a Generative
Latent Prior (GLP) — a flow-matching diffusion "meta-model" — on the residual-stream
activations of *benign* LLM conversations, then flag prompts whose activations are
off-manifold (high reconstruction error / diffusion-time). The proof-of-concept reused an
off-the-shelf GLP; the Master-Doc roadmap (**Phase 1**) now calls for training GLP on a
purpose-built benign corpus (WildChat, LMSYS-Chat-1M, …) and validating against adversarial
sets (WildJailbreak, WildGuardMix).

**The bottleneck for Phase 1 is data infrastructure.** Before we can train any GLP we need a
tool that, given *(a dataset, a pretrained open-source model, a set of layers)*, extracts the
residual-stream activations of every dataset entry and stores them in the exact on-disk format
the GLP trainer consumes — at scale (billion-token), with running normalization statistics and
self-documenting metadata. That tool is the **dataset manager** this plan designs.

It must also satisfy the standing action item *"Documentare datasets: come sono costruiti,
numero di esempi, formato"* — every produced dataset carries a manifest describing exactly how
it was built.

### Decisions locked with the user
- **Backend:** implement the high-throughput **vLLM/nnsight** async producer–consumer pipeline
  now (Phase-1 milestone), *but* behind a pluggable interface with the existing baukit
  `save_acts` path kept as a CPU-testable reference backend.
- **Granularity:** support **both** per-prompt pooled (`last` / `mean`) **and** per-token
  (`all`) activations. (GLP is a single-token model, so per-token simply yields many
  independent `(D,)` samples — no model change needed.)
- **Ingestion:** a **generic, config-driven** HF-`datasets` loader (text field *or* chat
  template, simple filters, dedup, sharding). Dataset-specific cleaning recipes (WildChat
  toxicity filtering, etc.) are added later as config files, not hard-coded now.

## Scope & self-containment

**`../generative_latent_prior/` is reference-only.** It is an example repo: nothing in
`guard-glp` may import from it at runtime, and any element we need (dataset handlers/loaders,
model code, …) is **ported into guard-glp's own `glp` package** rather than imported.

**Scope of this plan: dataset *creation* (the manager) + the *loader*** — enough to make the
**full round-trip testable**: write an activation dataset, then read it back through the *real*
consumer code in-repo. The GLP **training loop** (`glp_train.py` `main()`, `TrainConfig`, LR
schedulers, checkpointing) is **out of scope / a follow-up**.

### What already exists in guard-glp `src/glp/` (reuse in place — no porting)
- `save_acts` — already supports `last`/`mean`/`all` with masked mean-pooling
  (`src/glp/utils_acts.py:17`).
- `MemmapWriter` / `MemmapReader` (`src/glp/utils_acts.py:100`, `:153`). Note: `MemmapWriter`
  writes `data_*.npy` + `data_indices.npy` only — it does **not** write `dtype.txt`.
- `Normalizer` with `save_config` / `from_config` writing `rep_statistics.pt`, plus
  `GLP` / `Denoiser` (`src/glp/denoiser.py`) and `flow_matching.py`.
- HF dataset loaders `SourceHFDataset` / `CombinedHFDataset` with chat-templating,
  tokenization, dedup and decontamination (`src/preprocessing.py`).

### What must be ported from the reference `glp_train.py` (the loader)
The consumer classes `ActDataset`, `ActivationCollator`, `load_activation_dataset` (reads
`dtype.txt`, builds a `MemmapReader`) and `get_activation_dataloader`
(`generative_latent_prior/glp_train.py:55-134`). They import only `glp.denoiser` /
`glp.utils_acts` — both already present — so they port in cleanly. The surrounding training
loop is **not** ported here.

## Consumer contract (what the in-repo loader expects)

The consumer is the **in-repo** loader `glp.dataset` (the `ActDataset` / `load_activation_dataset`
machinery ported from the reference `glp_train.py`). It loads a dataset directory via
`MemmapReader` + an `ActDataset`. A directory is valid iff it contains:

| File | Produced by | Notes |
|------|-------------|-------|
| `data_0000.npy`, `data_0001.npy`, … | `MemmapWriter` | flat memmap chunks; **each sample is a flattened `(D,)` vector** |
| `data_indices.npy` | `MemmapWriter.flush()` | `uint64` `(N, 3)` = `(file_idx, start, end)` |
| `dtype.txt` | **us (new)** | single line, e.g. `float32` / `bfloat16` — `load_activation_dataset` does `np.dtype(dtype.txt)` then builds the `MemmapReader` (ref `glp_train.py:106-118`) |
| `rep_statistics.pt` | **us (new)** | `torch.save({"mean": (D,), "var": (D,)})` for `Normalizer` |
| `manifest.json` | **us (new)** | provenance/documentation (ignored by the loader) |

Three facts we will reuse rather than reinvent (all already in `src/glp/`):
- `glp.utils_acts.MemmapWriter` already produces `data_*.npy` + `data_indices.npy`
  (`src/glp/utils_acts.py:100`) — but **not** `dtype.txt`, so we write that ourselves.
- `glp.denoiser.Normalizer.save_config(path)` already writes
  `path/rep_statistics.pt` as `{"mean", "var"}` (`src/glp/denoiser.py`). We build a
  `Normalizer(mean, var)` from streamed stats and call it — no new serialization code.
- The ported `ActDataset` supports a **bf16-as-int16 storage trick**: a sample stored as
  `int16` is reinterpreted via `.view(torch.bfloat16)` then upcast to float (ref
  `glp_train.py:83`). The manager's `dtype: bfloat16` path must match this on-disk encoding.

Multi-layer convention: `ActDataset` parses `layer_(\d+)` from the directory name to tag samples
(ref `glp_train.py:69`). So we emit **one subdir per layer**: `<base>/layer_<NN>/…`. Single-layer
consumption points the loader at the specific `layer_<NN>` subdir.

## Goals / Non-goals

**Goals**
- One command turns *(HF dataset spec + model + layers + granularity)* into one
  trainer-ready dataset directory per layer, with `dtype.txt`, `rep_statistics.pt`, `manifest.json`.
- Streaming throughout: never hold the whole corpus in RAM; bounded-buffer producer→consumer.
- Online (single-pass) normalization statistics, numerically stable, mergeable across GPU shards.
- Multi-GPU sharding following the repo's existing `fire` + `.sh` convention.
- `dtype` choice (`float32` / `bfloat16`) to control disk footprint at billion-token scale.

**Non-goals (v1)**
- No changes to the GLP model / trainer (lives in a separate repo).
- No dataset-specific sanitization recipes baked in (config-driven generic filters only).
- No multi-token (joint sequence) modeling — out of scope for GLP itself.

## Architecture

```
                 ┌─────────────────────────── one process per GPU (shard) ──────────────────────────┐
 dataset spec ─► load_texts() ─► text shard ─►  PRODUCER  ──(bounded queue)──►  CONSUMER  ─► shard_<g>/
 (HF datasets)   field/chat-template          ExtractionBackend             Sink:               data_*.npy
                 filter/dedup/max_samples      .iter_batches() yields        - granularity pool   data_indices.npy
                                               (B,L,S,D)+mask per batch      - RunningStats       stats_partial.pt
                                                                             - MemmapWriter(/layer)
                 └──────────────────────────────────────────────────────────────────────────────────┘
                                                          │ all shards done
                                                          ▼
                          finalize():  merge shard_*/ ─► <base>/layer_<NN>/{data_*.npy, data_indices.npy}
                                        combine RunningStats ─► rep_statistics.pt  (via Normalizer.save_config)
                                        write dtype.txt + manifest.json
```

**Producer–consumer** (the Phase-1 "fixed-size buffer flushed once consumed"): the backend
produces activation batches into a `queue.Queue(maxsize=K)` (backpressure); a consumer thread
pools to the requested granularity, updates `RunningStats`, and writes via `MemmapWriter`.
Threads (not processes) so the consumer's numpy/memmap IO overlaps GPU work without CUDA
re-init pitfalls.

### Pluggable extraction backend
```python
class ExtractionBackend(Protocol):
    def iter_batches(self, texts: list[str]) -> Iterator[BatchActs]: ...
    #   BatchActs = (acts: Tensor[B,L,S,D], attention_mask: Tensor[B,S])
```
- **`HFBaukitBackend`** — wraps a refactored core of the existing `save_acts`
  (`src/glp/utils_acts.py:17`). Default; runs on CPU/single GPU; **no `serve` extra**; the
  basis for all unit tests.
- **`VLLMNNSightBackend`** — high-throughput path (requires `serve` extra: `vllm==0.9.2`,
  `nnsight==0.5.0`). Wraps the model with nnsight's vLLM integration, traces
  `model.layers[L].output` for the configured layers during generation/prefill, and yields
  per-layer hidden states. (Exact nnsight↔vLLM tracing API to be verified during
  implementation; the `BatchActs` contract isolates this risk.)

Both backends share one tracedict/layer-selection config and feed the identical consumer.

## Files to create / modify

**New package `src/glp/dataset/`** (cohesive, keeps `utils_acts.py` focused):
- `src/glp/dataset/__init__.py` — exports `build_shard`, `finalize`, `load_texts`, and the
  ported loader (`load_activation_dataset`, `get_activation_dataloader`, `ActDataset`).
- `src/glp/dataset/loader.py` — `load_texts(cfg, gpu_id, num_gpus) -> list[str]`:
  loads HF dataset (`path/name/split/revision`), selects `text_field` **or** applies
  `tokenizer.apply_chat_template` over a `conversation_field` (WildChat/LMSYS style), applies
  optional filters (`column == value`, min/max char/token length), optional dedup (hash set),
  `max_samples`, then shards `[gpu_id::num_gpus]`. **Reuse/extend the existing
  `src/preprocessing.py`** (`SourceHFDataset` / `CombinedHFDataset`, plus its chat-template,
  tokenization, dedup and decontamination helpers) instead of reimplementing HF ingestion.
- `src/glp/dataset/act_dataset.py` — **port of the loader/consumer** from
  `generative_latent_prior/glp_train.py:55-134`: `ActDataset`, `ActivationCollator`,
  `load_activation_dataset` (reads `dtype.txt` → `MemmapReader`), `get_activation_dataloader`.
  Imports only `glp.denoiser.Normalizer` and `glp.utils_acts.MemmapReader` (both in-repo).
  Adapt to repo conventions: strict pyright types (no `Optional`/`List`, **no `# type: ignore`,
  no hand-written stubs** per the standing typing rule), module-level
  `logger = logging.getLogger(__name__)`. The training loop (`main()`, `TrainConfig`, the
  `*_scheduler*` functions, `save_checkpoint`) is **not** ported here.
- `src/glp/dataset/backends.py` — `ExtractionBackend` protocol, `HFBaukitBackend`,
  `VLLMNNSightBackend`, and `make_backend(cfg)` factory.
- `src/glp/dataset/stats.py` — `RunningStats` (Chan/Welford parallel mean & M2, per layer);
  `.update(x)`, `.merge(other)`, `.save_partial(path)`, `RunningStats.load(path)`,
  `.to_normalizer() -> Normalizer`. Numerically stable; supports `bfloat16` inputs by
  accumulating in `float64`.
- `src/glp/dataset/builder.py` — `BuildConfig` dataclass; `build_shard(cfg, gpu_id)` (pass 1:
  loop backend→pool→stats→`MemmapWriter` per layer into `shard_<g>/layer_<NN>/`);
  `finalize(cfg)` (pass 2: merge shards, write `dtype.txt`, `rep_statistics.pt` via
  `Normalizer.save_config`, `manifest.json`).
- `src/glp/dataset/manifest.py` — assemble manifest (source dataset id/split/revision, model,
  layer, `retain`, granularity, num_samples, dim, dtype, max_length, filters, git SHA, UTC
  timestamp). Satisfies the "document datasets" action item.

**Refactor (small, signature-preserving):**
- `src/glp/utils_acts.py` — guard-glp's `save_acts` (`:17`) **already** handles `last`/`mean`/`all`
  with masked mean-pooling, so the refactor is narrow: its final `torch.cat(ret, dim=0)` (`:97`)
  cannot concatenate the ragged per-batch `(B,L,S,D)` tensors that `all` produces across batches
  with different sequence length `S`. Extract the per-batch trace loop into a generator
  `iter_activations(...) -> Iterator[tuple[Tensor, Tensor]]` (acts `(B,L,S,D)`, attention mask) so
  the consumer can pool/flatten per batch, fixing the ragged-`all` case. Reimplement `save_acts`
  on top of the generator so its public behavior/signature is unchanged (no existing tests touch
  its internals); `HFBaukitBackend` consumes the same generator.

**Scripts (follow existing `fire` + `.sh` GPU-sharding pattern, e.g. `eval_linear_probe`):**
- `scripts/dataset/build_activations.py` — `fire.Fire({"run": run, "finalize": finalize})`;
  `run(config, gpu_id)` loads YAML via OmegaConf and calls `build_shard`; `finalize(config)`
  calls `builder.finalize`.
- `scripts/dataset/build_activations.sh` — launches one `run` per GPU in parallel
  (`for gpu_id in …`), `wait`, then a single `finalize`. Mirrors `eval_linear_probe.sh`.

**Configs:**
- `configs/dataset/build_wildchat_llama8b_layer24.yaml` — the Phase-1 milestone target
  (Llama-3-8B-Instruct, layer 24, chat-templated, `last` + `all`).
- `configs/dataset/build_fineweb_llama1b_layer07.yaml` — small reproduction of the existing
  `llama1b-layer07` dataset for parity testing.

**Tests:**
- `tests/test_dataset.py` (CPU-only, tiny HF model e.g. `hf-internal-testing/tiny-random-...`
  or a stub): `RunningStats` mean/var correctness + shard-merge equals single-pass numpy
  `mean`/`var`; `build_shard`+`finalize` produces a dir with correct `dtype.txt`,
  `rep_statistics.pt` shapes, and `manifest.json`; pooled vs per-token sample counts. Uses
  `HFBaukitBackend` only (no `serve` extra).
- **Full round-trip test** (the point of porting the loader): `build_shard`+`finalize` →
  `load_activation_dataset(<base>/<gran>/layer_<NN>)` → `get_activation_dataloader(...,
  normalizer=Normalizer.from_config(rep_statistics.pt))` → assert batches load, `latents` are
  `(B, 1, D)`, normalization runs, and `layer_idx` is parsed from the dir name. Also covers the
  `bfloat16` (int16-encoded) path so the manager's encoding and `ActDataset.view(torch.bfloat16)`
  agree.

## Example config (sketch)

```yaml
# configs/dataset/build_wildchat_llama8b_layer24.yaml
save_root: .
model_name: meta-llama/Meta-Llama-3.1-8B-Instruct
output_dir: ${save_root}/data/llama8b-layer24-wildchat
backend: vllm_nnsight          # or: hf_baukit
num_gpus: 8
dataset:
  path: allenai/WildChat-1M
  split: train
  format: chat                 # chat -> apply_chat_template; text -> use text_field
  conversation_field: conversation
  text_field: null
  filters:                     # generic, optional
    - {column: language, equals: english}
  dedup: true
  max_samples: 1000000
extract:
  layer_prefix: model.layers
  layers: [24]
  retain: output
  granularity: [last, all]     # any of last|mean|all -> one dataset dir each, per layer
  max_length: 2048
  batch_size: 64
  dtype: bfloat16              # disk footprint; trainer reads via dtype.txt
  queue_maxsize: 16            # bounded producer buffer
  file_size: 33554432          # elements per memmap chunk
```

Output (per granularity × layer):
`data/llama8b-layer24-wildchat/last/layer_24/{data_*.npy, data_indices.npy, dtype.txt, rep_statistics.pt, manifest.json}`

## Verification (end-to-end)

1. **Static checks:** `make check` (ruff + strict pyright) and `make test` (pytest+coverage).
   New code must pass strict typing; follow repo logging (`logger = logging.getLogger(__name__)`).
2. **CPU smoke (no GPU, no `serve` extra):**
   `python scripts/dataset/build_activations.py run --config=configs/dataset/build_fineweb_llama1b_layer07.yaml --gpu_id=0`
   with `backend: hf_baukit`, a tiny model, and `max_samples: 256`; then `finalize`.
3. **Consumer round-trip (in-repo):** open the produced `layer_<NN>/` dir with the ported
   `glp.dataset.load_activation_dataset` and iterate `get_activation_dataloader(...)`; assert every
   sample is `(D,)`, `len(reader)==num_samples`, `rep_statistics.pt` loads via
   `Normalizer.from_config`, `dtype.txt` parses, and batches normalize cleanly. No dependency on
   `generative_latent_prior`. (Actual GLP training is a follow-up, out of scope here.)
4. **Stats correctness (test):** assert streamed+merged `mean`/`var` ≈ numpy single-pass over
   the same activations within tolerance.
5. **Scale path (GPU node, manual):** run `scripts/dataset/build_activations.sh` with
   `backend: vllm_nnsight` on a multi-GPU node against a real slice of WildChat; confirm shards
   merge and throughput is acceptable.

## Dependencies & risks
- **No imports from `generative_latent_prior`** — it is reference-only. The loader is *ported*
  into `src/glp/dataset/`, not imported; `make check` must pass with the sibling repo absent.
- **No new core deps** — `datasets`, `torch`, `transformers`, `numpy`, `omegaconf`, `fire`,
  `tqdm`, `baukit` are present. The vLLM/nnsight backend uses the **existing optional `serve`
  extra** (`vllm==0.9.2`, `nnsight==0.5.0`); guard its import so the package works without it.
- **Risk — nnsight↔vLLM tracing API:** exact call to capture per-layer hidden states during
  vLLM generation must be verified against `nnsight==0.5.0`; the `BatchActs` contract isolates
  it, and `HFBaukitBackend` is a correctness oracle.
- **Risk — multi-layer normalizer indexing:** the trainer's `Normalizer.get_layer_stat` indexes
  by absolute layer id; v1 sidesteps this by emitting independent per-layer dirs with per-layer
  `(D,)` stats (matches the existing single-layer `train_*` configs). Joint multi-layer training
  is a downstream concern, flagged for follow-up.
- **Future extension:** a third granularity (token subsampling), more filter predicates, and a
  resumable/append mode — all fit behind the current interfaces without redesign.
