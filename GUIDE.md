# Guard-GLP — Usage Guide

How to install the project and run each component. All commands run **from the repo
root** with the virtual environment active. The package installs as `guard-glp` but
**imports as `glp`** (src layout).

- [Installation](#installation)
- [Dataset manager](#dataset-manager) · [GLP training](#glp-training) · [Detection](#detection) ·
  [Steering](#steering) · [Inference / judging](#inference--judging) ·
  [Visualization](#visualization) · [Preprocessing](#preprocessing) ·
  [GLP weights & library use](#glp-weights--library-use) · [Development](#development)

---

## Installation

Requires Python 3.12 with [uv](https://docs.astral.sh/uv/). Two sync modes, both creating
`.venv/` from the locked deps in `uv.lock`:

```bash
uv sync                       # core/dev: detection, steering, visualization, hf_baukit
                              #           dataset builds, GLP training, and code checks
uv sync --extra serve         # + the vLLM serving stack (see below)
source .venv/bin/activate     # or prefix commands with `uv run`
```

### vLLM serving stack (optional, cluster-only)

`uv sync --extra serve` adds the `serve` extra, needed by the LLM-judge server
(`scripts/inference/serve_llm.sh`) and the `vllm_nnsight` dataset backend. The extra is
just **`vllm`**; `nnsight` is already a core dependency (installed by plain `uv sync`), as
are `baukit`/`transformers`. This is the `.venv` the SLURM dataset workers activate, so
build it with `--extra serve` before running the dataset manager with
`backend: vllm_nnsight`.

vllm/nnsight have no macOS wheels and a fragile install order. `uv sync --extra serve`
resolves it via the lockfile; if you must install by hand, use a dedicated env in this
exact sequence, ignoring pip warnings:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install vllm==0.9.2
uv pip install transformers==4.47.0   # NB: transformers is unpinned in pyproject.toml —
uv pip install -e .                    #     uv.lock pins the resolved version for `uv sync`
```

### Environment & auth (`.env`)

Configuration lives in a `.env` (gitignored). Copy the template and fill it in:

```bash
cp .env.example .env
```

It is loaded automatically — the dataset shell workers `source .env`, and the Python
entry points call `load_dotenv()`. Variables:

| Variable | Purpose |
|----------|---------|
| `HF_HOME`, `UV_CACHE_DIR` | HuggingFace / uv cache dirs. Point at the **shared filesystem** (see below). |
| `HF_TOKEN` | Gated models (`meta-llama/*`) and Hub pushes. Also run `huggingface-cli login` and accept each model's license on the Hub. |

**Cluster filesystem layout.** The repo, `.venv`, `logs/`, and dataset `output_dir`s
live on the **across-node shared filesystem**. Put `HF_HOME`/`UV_CACHE_DIR` there too, so
model weights download **once and are reused by every node** (per-node `$SCRATCH` would
re-download on each node). `output_dir` must never be on per-node `$SCRATCH`, or a SLURM
shard on one node and the finalize on another won't see the same files.

---

## Dataset manager

`scripts/dataset/` + `src/glp/dataset/`. Turns *(HF dataset + model + layers + granularity)*
into trainer-ready activation datasets. Two passes: **extract** per-shard, then
**finalize** (merge shards, write normalization stats + manifest).

**Output layout** (one dir per granularity × layer):

```
<output_dir>/<granularity>/layer_<NN>/
  data_0000.npy …        # flat memmap chunks; one (D,) vector per sample
  data_indices.npy       # (N, 3) uint64 (file_idx, start, end)
  dtype.txt              # "float32" or "int16" (bfloat16 stored as int16)
  rep_statistics.pt      # {"mean": (D,), "var": (D,)} for the Normalizer
  manifest.json          # provenance: source, model, layer, counts, git SHA, ...
```

### Token-count pass (before extraction)

To size an extraction (and see how many prompts would be truncated at
`extract.max_length`), count tokens with the model's tokenizer first. This reports stats
for **both** prompt views regardless of the config's own `prompt_view`:

```bash
python scripts/dataset/count_tokens.py run \
    --config=configs/dataset/build_guardglpbenign_llama1b_last_useronly.yaml
```

Per view (`user`, `full`) it prints `n_prompts`, `total_tokens`, mean / median / p90 / p99,
and `n_over_max_length`. It builds the exact prompt strings extraction will feed (reusing
`glp.dataset.loader.load_texts`), so the counts match the real run. CPU-only.

### Local / CPU (quickstart, no SLURM, no `serve` extra)

Use `hf_baukit` with the small smoke config (gsm8k, 256 samples, Llama-3.2-1B — a quick,
cached download; needs the model license accepted, see `.env` above). Run the two passes
directly; `--device=cpu` forces CPU (it also auto-falls back when no GPU is visible):

```bash
python scripts/dataset/build_activations.py run \
    --config=configs/dataset/build_smoke_gsm8k_llama1b_baukit.yaml --gpu_id=0 --device=cpu
python scripts/dataset/build_activations.py finalize \
    --config=configs/dataset/build_smoke_gsm8k_llama1b_baukit.yaml
```

`run` extracts one shard (`gpu_id` = shard index); `finalize` merges all shards. (The
`build_fineweb_llama1b_layer07.yaml` config reproduces the reference llama1b-layer07
dataset but downloads the full FineWeb `sample-10BT` — large; not a quickstart.)

### SLURM (default cluster mechanism)

One command submits the whole pipeline — a GPU **job array** for extraction and a
**CPU-only** finalize that runs (via `--dependency=afterok`) only after every shard
succeeds, so no GPU is held during the merge:

```bash
bash scripts/dataset/build_activations.sh configs/dataset/build_wildchat_llama8b_layer24.yaml
# -> prints the pass-1 and pass-2 job IDs; track with `squeue --me`
```

Partition and GPU constraint are cluster-specific. The orchestrator defaults to
partition `defq` and no constraint; override per cluster without editing files:

```bash
GLP_PARTITION=long GLP_CONSTRAINT='ampere|lovelace|hopper' \
    bash scripts/dataset/build_activations.sh CONFIG
```

The orchestrator picks the topology from `extract.tensor_parallel_size` (TP), not the
backend:

| `tensor_parallel_size` | Pass 1 submission | `num_gpus` | When |
|------------------------|-------------------|-----------|------|
| `1` (default) | job array, **1 GPU per task** — data-parallel shards (one full model per GPU, striding the corpus) | # shards | model fits on one GPU; both backends |
| `N > 1` | a **single** task with N GPUs (one shard, tensor parallel) | 1 | model too large for one GPU; `vllm_nnsight` only |

Data-parallel is preferred whenever the model fits on a single GPU (e.g. Llama-3.1-8B on
an 80 GB H100): higher throughput, no cross-GPU communication, and per-shard resumability.

Logs land in `logs/build_shard_%A_%a.{out,err}` and `logs/build_finalize_%j.{out,err}`.
Re-run a single failed shard directly:

```bash
sbatch --array=3 --gres=gpu:1 scripts/dataset/_run_shard.sbatch CONFIG
```

### Consuming a dataset

The in-repo loader reads a `layer_<NN>/` dir back for training/eval:

```python
from glp.dataset import get_activation_dataloader, load_activation_dataset
from glp.denoiser import Normalizer

layer_dir = "data/llama1b-layer07-fineweb/last/layer_07"
dataset = load_activation_dataset(layer_dir)                       # reads dtype.txt
normalizer = Normalizer.from_config(f"{layer_dir}/rep_statistics.pt")
loader = get_activation_dataloader(dataset, batch_size=4096, normalizer=normalizer)
for batch in loader:
    latents = batch["latents"]   # (B, 1, D), normalized; layer_idx parsed from dir name
    break
```

### Config reference

```yaml
model_name: meta-llama/Meta-Llama-3.1-8B-Instruct
output_dir: ${save_root}/data/llama8b-layer24-wildchat   # must be on the shared FS
backend: vllm_nnsight        # or hf_baukit
num_gpus: 8                  # data-parallel shards: one full model per GPU (see topology table)
dataset:                     # swap this whole block to target a different / custom HF dataset
  path: allenai/WildChat-1M
  split: train               # supports HF slice syntax, e.g. train[:5000] for a quick test
  format: chat               # chat -> apply_chat_template(conversation_field); text -> text_field
  conversation_field: conversation
  prompt_view: full          # chat-only. full = whole conversation (add_generation_prompt=False);
                             # user = first user turn as a standalone single-turn prompt
                             # (add_generation_prompt=True) — the deployment screening position.
  filters: [{column: language, equals: English}]   # optional column == value
  dedup: true
  max_samples: 1000000       # global cap, applied before sharding
extract:
  layers: [24]
  retain: output             # capture layer input | output
  granularity: [last]        # last | mean | all -> one dataset dir each, per layer
  dtype: bfloat16            # float32 | bfloat16 (stored as int16)
  batch_size: 16             # vllm_nnsight prefills the whole batch in one pass; keep modest
  max_length: 2048
  add_special_tokens: null   # null -> auto (chat: no extra BOS; text: add BOS); set true/false to override
  tensor_parallel_size: 1    # 1 = data-parallel (preferred); >1 = tensor parallel for a model too big for one GPU
```

The full WildChat-1M → Llama-3.1-8B layer-24 campaign is just the shipped config:

```bash
bash scripts/dataset/build_activations.sh configs/dataset/build_wildchat_llama8b_layer24.yaml
```

**GLP training-corpus builds.** The two shipped `guard-glp-benign` configs extract the
GLP training activations from `ddidacus/guard-glp-benign` (assembled by
`merge_train_sets.py`, see [Preprocessing](#preprocessing)) with Llama-3.2-1B-Instruct at
layers `[8, 12, 14]` (50% / 75% / second-to-last), last-token, one per prompt view:

```bash
bash scripts/dataset/build_activations.sh configs/dataset/build_guardglpbenign_llama1b_last_useronly.yaml
bash scripts/dataset/build_activations.sh configs/dataset/build_guardglpbenign_llama1b_last_full.yaml
# -> data/llama1b-guardglpbenign-{useronly,full}/last/layer_{08,12,14}/
```

---

## GLP training

`scripts/train/` + `src/glp/train/`. Trains a flow-matching diffusion GLP
(`glp.denoiser.GLP`) on one built `layer_<NN>/` activation dir. The model architecture and
the training data are fully **config-driven** (`configs/train/`); the entry point is an
OmegaConf CLI (`config=<file>` plus `key=value` overrides).

### Single run

```bash
python scripts/train/train_glp.py \
    config=configs/train/glp_llama1b_guardglpbenign.yaml \
    view=useronly layer=08 device=cuda:0
```

`view` (`useronly|full`) and `layer` (`08|12|14`) are selectors: they resolve
`train_dataset=data/llama1b-guardglpbenign-<view>/last/layer_<NN>` and its
`rep_statistics.pt`. Trains on GPU by default (`device=cuda:0`; pass `device=cpu` for a
quick smoke run).

### All six runs (guard-glp-benign)

One GLP per prompt-view × layer (`{useronly,full} × {08,12,14}`):

```bash
bash scripts/train/train_glp_all.sh            # or: ... [device]
```

### Overrides & other datasets

Override any field on the CLI — point at a different built dataset or change the
architecture without editing the config:

```bash
python scripts/train/train_glp.py config=configs/train/glp_llama1b_guardglpbenign.yaml \
    train_dataset=data/my-acts/last/layer_12 rep_statistic=data/my-acts/last/layer_12/rep_statistics.pt \
    output_path=runs/my-glp batch_size=2048 glp_kwargs.denoiser_config.n_layers=6
```

The reference config `configs/train_llama1b_static.yaml` (FineWeb, base
`meta-llama/Llama-3.2-1B`) reproduces the paper's single-layer GLP.

### Output layout & reloading

Each run writes to `runs/<run_name>/`:

```
final.safetensors                 # latest GLP weights (denoiser)
rep_statistics.pt                 # normalizer mean/var
config.yaml                       # fully-resolved run config
checkpoints/epoch_<N>.safetensors # per-epoch snapshots (save_epochs)
optimizer_state.pt                # if save_opt_state: true
```

Reload a local checkpoint:

```python
from omegaconf import OmegaConf
from glp.denoiser import GLP

run = "runs/glp-llama1b-guardglpbenign-useronly-layer08-d3"
model = GLP(**OmegaConf.load(f"{run}/config.yaml").glp_kwargs)
model.load_pretrained(run, name="final")
```

(`glp.denoiser.load_glp(...)` is the loader for Hub-hosted checkpoints — see
[GLP weights & library use](#glp-weights--library-use).)

### Config reference

```yaml
model_name: meta-llama/Llama-3.2-1B-Instruct
train_dataset: ${save_root}/data/llama1b-guardglpbenign-${view}/last/layer_${layer}
rep_statistic: ${train_dataset}/rep_statistics.pt
output_path: ${save_root}/runs/${run_name}
num_epochs: 1
glp_kwargs:                    # architecture + normalizer (swap to change model size)
  normalizer_config: {rep_statistic: ${rep_statistic}}
  denoiser_config:
    d_input: 2048              # LLM hidden dim
    d_model: 4096              # d_input * 2
    d_mlp: 8192                # d_input * 4
    n_layers: 3                # GLP depth (not the LLM layer being modeled)
    multi_layer_n_layers: null # int -> single layer-conditioned GLP over multiple LLM layers
use_bf16: true
learning_rate: 5e-5
batch_size: 4096
lr_scheduler: {scheduler_cls: cosine_scheduler_with_warmup, warmup_ratio: 0.01,
               initial_factor: 0.01, final_factor: 0.1}
wandb_enabled: false           # true -> requires the optional `wandb` package
```

No dedicated training `.sbatch` yet — launch on SLURM via `srun`, e.g.:

```bash
srun --partition=defq --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=8:00:00 \
    python scripts/train/train_glp.py config=configs/train/glp_llama1b_guardglpbenign.yaml \
    view=useronly layer=08 device=cuda:0
```

---

## Detection

`scripts/detection/` — Guard-GLP anomaly classifiers and baselines. Two-pass workflow:
parallel per-GPU `run`, then single-process `aggregate` into `results.json`.

```bash
# GLP detectors (one config each, same launcher)
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_reconstruction_err.yaml
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_pi.yaml        # path-integral / Hutchinson
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_dte.yaml       # diffusion-time estimation
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_dte_glp.yaml   # DTE w/ GLP-sampled reference

# Baselines
sbatch scripts/detection/eval_linear_probe.sh configs/detection/eval_lp.yaml      # supervised probe
sbatch scripts/detection/eval_diffmean.sh    configs/detection/eval_diffmean.yaml # difference-of-means

# LP + PI + DTE + GLP-DTE + DiffMean sequentially, or OOD generalization
bash  scripts/detection/eval_all.sh
sbatch scripts/detection/eval_ood.sh
```

(`eval_all.sh` covers LP/PI/DTE/GLP-DTE/DiffMean; reconstruction-error is run separately via
the `eval_classifier.sh configs/detection/eval_reconstruction_err.yaml` line above.)

Each writes `results.json` (per-layer + aggregate AUPRC, thresholds) and PR/threshold
plots to the config's `out_dir`. Run a pass manually, e.g.:

```bash
python scripts/detection/evaluate_classifier.py run --config=CONFIG --gpu_id=0
python scripts/detection/evaluate_classifier.py aggregate --out_dir=results/<name>
```

---

## Steering

`scripts/steering/` — GLP-regularized activation steering on adversarial prompts, judged
for safety by an LLM judge served via vLLM (model id set in
`scripts/inference/serve_llm.sh`). Two passes (`run` per GPU → `aggregate`), driven by env
vars (`STEERING_TYPE` ∈ `none|sv|glp`, `ALPHAS`, `OUT_DIR`; `NUM_GPUS`, `MODEL` also
overridable).

```bash
bash scripts/steering/run_steering_benign.sh    # benign/malicious, all variants
bash scripts/steering/run_steering_refusal.sh   # refusal/compliance

# custom run
STEERING_TYPE=glp ALPHAS="-0.1,0.1,1.0" OUT_DIR=results/steering_glp \
    bash scripts/steering/steering.sh
```

Configs live in `configs/paper/steering/`.

---

## Inference / judging

`scripts/inference/` — serve an LLM judge with vLLM (needs the `serve` extra), then
(re-)judge generated responses.

```bash
sbatch scripts/inference/serve_llm.sh                      # serve the judge (vLLM)
RESULTS_DIR=results/steering_glp bash scripts/inference/rejudge.sh   # re-judge one dir
bash scripts/inference/rejudge_all.sh                      # re-judge all steering results
```

`rejudge_responses.py` reads `*responses*.json` in `RESULTS_DIR` and writes `rejudged_*.json`.

---

## Visualization

`scripts/visualization/` — PCA / t-SNE projections of activations and GLP reconstructions.

```bash
sbatch scripts/visualization/visualize_activations.sh      # extract (per-GPU) + plot

# or manually
python scripts/visualization/visualize_activations.py --config=configs/visualization/eval_plotting.yaml --gpu_id=0
python scripts/visualization/visualize_activations.py --aggregate --results_dir=results/<name> --layers="1,7,15" --method=pca
```

---

## Preprocessing

`scripts/preprocessing/merge_train_sets.py` — assemble the benign training corpus
(LMSYS-Chat-1M, WildChat, WildChat-4.8M, WildGuardMix): sanitize, dedup, decontaminate
against WildJailbreak.

```bash
python scripts/preprocessing/merge_train_sets.py --output_dir ./data/guard-glp-benign
python scripts/preprocessing/merge_train_sets.py --push_to_hub --repo_id ddidacus/guard-glp-benign --private
```

Key flags: `--output_dir`, `--push_to_hub`, `--repo_id`, `--private`, `--sim_threshold`
(decontamination cosine threshold, default `0.95`).

---

## GLP weights & library use

Pretrained GLP checkpoints load directly (public, no auth):

```python
from glp.denoiser import load_glp
model = load_glp("generative-latent-prior/glp-llama1b-d12-multi", device="cuda:0", checkpoint="final")
```

The `-dN` suffix is the GLP depth (`n_layers`) and `-multi` marks a multi-layer
(layer-conditioned) GLP; the detection pipeline uses `glp-llama1b-d12-multi` for Llama-1B and
`glp-llama8b-d6` for Llama-8B.

---

## Development

Code quality (ruff + strict pyright) and tests, mirrored from CI:

```bash
make check     # ruff check + ruff format --check + pyright
make test      # pytest with coverage (slow/GPU/network tests deselected by default)
make format    # auto-format
make all       # check + test
```
