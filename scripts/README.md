# scripts/

Pipeline for building the safety dataset, visualising activations, and evaluating guardrail classifiers. All scripts are meant to be run **from the project root** with the virtual environment activated.

```bash
export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
source .venv/bin/activate
```

---

## 1. Dataset

### Option A — use the published dataset directly

The dataset `ddidacus/guard-glp-data` is already on the Hub and is downloaded automatically by every evaluation script. No local preparation is needed.

| Split | Size | Benign | Malicious |
|-------|-----:|-------:|----------:|
| train | ~7 700 | ~3 500 | ~4 200 |
| calibration | ~1 100 | ~500 | ~600 |
| test | ~2 200 | ~1 000 | ~1 200 |

Fields: `prompt` (string), `adversarial` (bool — `True` = malicious).

### Option B — rebuild

`scripts/create_dataset.py` pulls from three upstream sources (FineWeb, centrepourlasecuriteia/jailbreak-dataset, ddidacus/harmeval-gcg-llama3-1b) and assembles the three splits with a 70/10/20 random draw.

---

## 2. Visualise activations (t-SNE / PCA)

Extracts last-token residual-stream activations for the full training split across 4 GPUs, reconstructs each via the GLP, and produces per-layer scatter plots (original / reconstructed / reconstruction error) using t-SNE or PCA.

**Config:** `configs/paper/eval_plotting.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `"1b"` | `"1b"` (Llama-3.2-1B) or `"8b"` (Llama-3.1-8B) |
| `layers` | `[1, 7, 11, 15]` | Residual-stream layer indices to probe |
| `noise_level` | `0.0` | Flow-matching noise level before reconstruction |
| `num_timesteps` | `200` | Denoising steps |
| `method` | `tsne` | Dimensionality reduction: `pca` or `tsne` |
| `out_dir` | `results/paper-viz-tsne` | Output directory |
| `num_gpus` | `4` | Number of parallel GPU workers |

**Submit (Slurm):**
```bash
sbatch scripts/visualize_activations.sh
```

**Run manually (two passes):**
```bash
# Pass 1 — extract shards in parallel
for gpu_id in 0 1 2 3; do
    python scripts/visualize_activations.py \
        --config=configs/paper/eval_plotting.yaml --gpu_id=$gpu_id &
done
wait

# Pass 2 — aggregate and plot
python scripts/visualize_activations.py --aggregate \
    --results_dir=results/paper-viz-tsne \
    --layers="1,7,11,15" \
    --method=tsne
```

**Outputs** (in `out_dir`):
- `{method}_scatter2d_layer{N}.png/.pdf` — 3-panel scatter per layer (one file per layer)
- `{method}_{N}.png` — 1-D component histograms per layer
- `aggregated_errors.png` — reconstruction-error histograms
- `error_by_layer.png` — mean error ± std across layers
- `reconstruction_error_gap.json` — numeric summary

---

## 3. Evaluate classifiers

All classifiers follow the same **two-pass** workflow:

1. **Pass 1** — activation / score extraction, one process per GPU, results written as shards (`*.th`) in `out_dir`.
2. **Pass 2** — aggregate shards, compute metrics, save `results.json` and plots.

The bash wrappers handle both passes and accept an optional config path as `$1`.

### 3a. GLP path-integral density (Hutchinson's estimator)

Estimates log p(x) via Hutchinson's stochastic trace estimator applied to the flow-matching vector field.

**Config:** `configs/paper/eval_pi.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `hutchinson` | Fixed for this evaluator |
| `num_hutchinson_samples` | `8` | Random vectors for trace estimation |
| `noise_level` | `0.0` | Starting noise level |
| `num_timesteps` | `200` | ODE integration steps |
| `layers` | `[7, 13, 15]` | Layers to score |
| `batch_size` | `128` | Inference batch size |
| `out_dir` | `results/paper-eval-glp-hutchinson` | |

```bash
sbatch scripts/evaluate_classifier.sh configs/paper/eval_pi.yaml
# or: bash scripts/evaluate_classifier.sh configs/paper/eval_pi.yaml
```

### 3b. GLP diffusion time estimation (DTE)

Classifies by the expected noise level σ at which the GLP assigns maximum likelihood to the observed activation, using a kNN reference set drawn from the training split.

**Config:** `configs/paper/eval_dte.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `dte` | Fixed for this evaluator |
| `reference_num_samples` | `4096` | kNN reference set size (from train benign) |
| `dte_K` | `8` | Neighbours for kNN density |
| `dte_num_sigma_bins` | `512` | Histogram bins for σ estimation |
| `out_dir` | `results/paper-eval-dte` | |

```bash
sbatch scripts/evaluate_classifier.sh configs/paper/eval_dte.yaml
```

### 3c. GLP-DTE (GLP-sampled reference)

Same as DTE but the kNN reference set is drawn by sampling the GLP generative model rather than from the training data.

**Config:** `configs/paper/eval_dte_glp.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `dte_glp` | Fixed for this evaluator |
| `glp_sample_steps` | `100` | Steps used to sample reference activations |
| `reference_num_samples` | `4096` | Number of GLP samples for reference |
| `dte_K` | `8` | |
| `dte_num_sigma_bins` | `1024` | |
| `out_dir` | `results/paper-eval-dte-glp` | |

```bash
sbatch scripts/evaluate_classifier.sh configs/paper/eval_dte_glp.yaml
```

### 3d. GLP reconstruction error

Scores each prompt by the L2 distance between the original activation and its GLP reconstruction after partial noising. Higher error = more anomalous = more likely malicious.

**Config:** `configs/paper/eval_reconstruction_err.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `method` | `reconstruction_error` | Fixed for this evaluator |
| `noise_level` | `0.0` | Noise added before denoising |
| `rec_num_timesteps` | `50` | Denoising steps for reconstruction |
| `out_dir` | `results/paper-eval-recon_err` | |

```bash
sbatch scripts/evaluate_classifier.sh configs/paper/eval_reconstruction_err.yaml
```

### 3e. Linear probe (supervised baseline)

Trains a one-layer linear classifier directly on the residual-stream activations using the train split, calibrates a threshold on the calibration split, and reports metrics on the test split. Also runs sanity checks (norm-AUROC, shuffled-label probe, length-stratified AUROC).

**Config:** `configs/paper/eval_lp.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `token_pooling` | `mean` | Token aggregation: `mean` or `last` |
| `probe_lr` | `1e-3` | AdamW learning rate |
| `probe_epochs` | `100` | Training epochs |
| `probe_wd` | `1e-4` | Weight decay |
| `probe_batch_size` | `64` | |
| `probe_device` | `cuda` | Device for probe training |
| `out_dir` | `results/paper-eval-lp` | |

```bash
sbatch scripts/eval_linear_probe.sh configs/paper/eval_lp.yaml
```

### 3f. DiffMean (baseline)

Computes a per-layer steering vector as the normalised difference of means between calibration benign and malicious activations. Scores prompts by their cosine similarity with the steering vector.

**Config:** `configs/paper/eval_diffmean.yaml`

| Key | Default | Description |
|-----|---------|-------------|
| `token_pooling` | `mean` | Token aggregation |
| `layers` | `[7, 13, 15]` | |
| `out_dir` | `results/paper-eval-diffmean` | |

```bash
sbatch scripts/eval_diffmean.sh configs/paper/eval_diffmean.yaml
```

---

## 4. Run all evaluations

```bash
bash scripts/evaluate_all.sh
```

This runs the linear probe, GLP-PI, GLP-DTE, GLP-DTE-GLP, and DiffMean evaluations sequentially using the paper configs.

---

## Output format

Every classifier writes a `results.json` to its `out_dir` with the following structure:

```
{
  "config": { ... },
  "per_layer": {
    "layer_7":  { "auprc": ..., "youden": { "tpr": ..., "fpr": ..., ... }, "tpr60": ..., ... },
    ...
  },
  "aggregate": {
    "mean":       { ... },   # score averaged across layers
    "min":        { ... },   # most anomalous layer wins
    "best_layer": { ... }    # single best layer by AUPRC
  }
}
```

Plots saved alongside `results.json`:
- `pr_curves.png` — precision-recall curves for all aggregation strategies
- `tpr_fnr_threshold.png` — TPR / FNR vs threshold for each strategy
