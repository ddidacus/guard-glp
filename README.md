# Guard-GLP: Generative Meta-Models are Adversarial Classifiers

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg?style=for-the-badge)](https://arxiv.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)](https://huggingface.co/datasets/ddidacus/guard-glp-data)

**Authors**: Diego Calanzone, Pietro Greiner. <br>
**Affiliation**: Mila Quebec AI Institute, Universite de Montreal.

We study adversarial prompt detection in LLMs through the geometry of intermediate activations. Our hypothesis is that token-level adversarial attacks induce hidden states that lie outside the activation manifold associated with typical prompts. To test this, we use Generative Latent Priors (GLP), diffusion-based meta-models trained on LLM residual stream activations, as proxies for this manifold. We evaluate several training-free anomaly scores derived from GLP, namely **Guard-GLP**, including reconstruction error, diffusion time estimation, and a density-based estimate using Hutchinson trace estimation. Empirically, GLP reconstruction error separates benign prompts from adversarial prompts and provides a competitive classifier without supervised fine-tuning. We also study GLP as a regularizer for activation steering, showing that denoising edited activations can reduce attack success while limiting over-refusal and nonsensical generations. Overall, our results suggest that generative priors over LLM activations provide a useful interface for both adversarial prompt detection and safer activation-level interventions.

## Compute

Most evaluation scripts require **<24 GB VRAM** (e.g. a single NVIDIA RTX 4090). Multi-GPU scripts (steering, visualisation) are configured for **4-8 GPUs** via Slurm and can be adjusted via environment variables or config files. The LLM judge server (`serve_llm.sh`) requires **4x 80 GB GPUs** (e.g. A100).

## Setup

This code was tested with Python 3.11. We build on the [GLP codebase](https://github.com/generative-latent-prior/glp) by Luo et al., 2026.

1. Create the conda environment and install dependencies:
```bash
conda env create -f environment.yaml
conda activate glp
pip install vllm==0.9.2
pip install transformers==4.47.0
pip install -e .
```
> **Note:** Install in this exact order and ignore pip warnings. This is the only combination that makes vllm/nnsight/transformers work together.

2. Set cache paths (required on the Mila cluster to avoid disk quota errors):
```bash
export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
```

## Pre-Trained Weights

We use the pre-trained GLP checkpoints from [generative-latent-prior](https://huggingface.co/generative-latent-prior). For quick loading:
```python
from glp.denoiser import load_glp
model = load_glp("generative-latent-prior/glp-llama1b-d6", device="cuda:0", checkpoint="final")
```

| Model | HuggingFace |
|-|-|
| GLP Llama-1B (d=6) | [Link](https://huggingface.co/generative-latent-prior/glp-llama1b-d6) |
| GLP Llama-8B (d=6) | [Link](https://huggingface.co/generative-latent-prior/glp-llama8b-d6) |

## Dataset

The dataset `ddidacus/guard-glp-data` is hosted on the HuggingFace Hub and is **downloaded automatically** by all evaluation scripts. No manual preparation is needed.

| Split | Size | Benign | Malicious |
|-------|-----:|-------:|----------:|
| train | ~7 700 | ~3 500 | ~4 200 |
| calibration | ~1 100 | ~500 | ~600 |
| test | ~2 200 | ~1 000 | ~1 200 |

Fields: `prompt` (string), `adversarial` (bool, `True` = malicious).

To rebuild the dataset from upstream sources (FineWeb, jailbreak-dataset, harmeval-gcg):
```bash
python scripts/create_dataset.py
```

---

## Reproducing the Experiments

All scripts should be run **from the project root** with the virtual environment activated.

### 1. Activation visualisation (t-SNE / PCA)

Extracts last-token residual-stream activations, reconstructs each via the GLP, and produces per-layer scatter plots.

**Config:** `configs/paper/eval_plotting.yaml`

```bash
# Slurm
sbatch scripts/visualize_activations.sh

# Manual (4 GPUs)
for gpu_id in 0 1 2 3; do
    python scripts/visualize_activations.py \
        --config=configs/paper/eval_plotting.yaml --gpu_id=$gpu_id &
done
wait

python scripts/visualize_activations.py --aggregate \
    --results_dir=results/paper-viz-tsne \
    --layers="1,7,11,15" --method=tsne
```

### 2. Adversarial prompt detection (Guard-GLP classifiers)

All classifiers follow a **two-pass** workflow: (1) parallel per-GPU activation/score extraction, (2) aggregation into `results.json` and plots.

#### 2a. GLP reconstruction error

Scores each prompt by the L2 distance between the original activation and its GLP reconstruction.

```bash
sbatch scripts/evaluate_classifier.sh configs/paper/eval_reconstruction_err.yaml
```

#### 2b. GLP path-integral density (Hutchinson's estimator)

Estimates log p(x) via Hutchinson's stochastic trace estimator on the flow-matching vector field.

```bash
sbatch scripts/evaluate_classifier.sh configs/paper/eval_pi.yaml
```

#### 2c. Diffusion time estimation (DTE)

Classifies by the expected noise level at which the GLP assigns maximum likelihood, using a kNN reference set from the training split.

```bash
sbatch scripts/evaluate_classifier.sh configs/paper/eval_dte.yaml
```

#### 2d. GLP-DTE (GLP-sampled reference)

Same as DTE, but the kNN reference set is drawn by sampling the GLP generative model.

```bash
sbatch scripts/evaluate_classifier.sh configs/paper/eval_dte_glp.yaml
```

#### 2e. Linear probe (supervised baseline)

Trains a one-layer linear classifier on residual-stream activations, calibrates a threshold, and reports metrics.

```bash
sbatch scripts/eval_linear_probe.sh configs/paper/eval_lp.yaml
```

#### 2f. DiffMean (baseline)

Computes a per-layer normalised difference-of-means steering vector and scores prompts by cosine similarity.

```bash
sbatch scripts/eval_diffmean.sh configs/paper/eval_diffmean.yaml
```

#### Run all evaluations at once

```bash
bash scripts/evaluate_all.sh
```

### 3. Activation steering

Evaluates GLP-regularised activation steering on adversarial prompts. Responses are judged for safety using Llama-Guard-3-8B.

#### 3a. Steering on adversarial (benign-malicious) prompts

```bash
# Steering types: none, sv (additive), glp (GLP manifold projection)
# Node 1 — runs all three steering variants
bash scripts/node1.sh
```

Or individually:
```bash
STEERING_TYPE=glp ALPHAS="0.01,0.1,0.5,1.0,2.0" OUT_DIR=results/steering_glp bash scripts/steering.sh
STEERING_TYPE=sv  ALPHAS="0.01,0.1,0.5,1.0,2.0" OUT_DIR=results/steering_sv  bash scripts/steering.sh
STEERING_TYPE=none OUT_DIR=results/steering_none bash scripts/steering.sh
```

#### 3b. Steering on refusal/compliance prompts

```bash
# Node 2 — runs all three steering variants for refusal
bash scripts/node2.sh
```

Or use the config-based approach:
```bash
python scripts/steering.py run --config=configs/paper/steering/refusal_glp.yaml --gpu_id=0
python scripts/steering.py aggregate --out_dir=results/steering_refusal_glp
```

Available steering configs:

| Config | Steering type | GLP regularised |
|--------|--------------|-----------------|
| `steering/classic.yaml` | refusal | No |
| `steering/glp.yaml` | refusal | Yes |
| `steering/compliance_glp.yaml` | compliance | Yes |
| `steering/compliance_noglp.yaml` | compliance | No |
| `steering/compliance_direct.yaml` | compliance | No |
| `steering/refusal_glp.yaml` | refusal | Yes |
| `steering/refusal_noglp.yaml` | refusal | No |
| `steering/nosteering_glp.yaml` | none | Yes |
| `steering/nosteering_noglp.yaml` | none | No |
| `steering/random.yaml` | random | No |

#### 3c. Re-judging responses

To re-judge previously generated responses with an LLM judge served via vLLM:

```bash
# Terminal 1: serve the judge model
sbatch scripts/serve_llm.sh

# Terminal 2: re-judge
RESULTS_DIR=results/steering_glp bash scripts/rejudge.sh
```

### 4. OOD evaluation

```bash
sbatch scripts/eval_ood.sh
```

---

## Output format

Every classifier writes a `results.json` to its `out_dir`:

```json
{
  "config": { "..." },
  "per_layer": {
    "layer_7":  { "auprc": 0.0, "youden": { "tpr": 0.0, "fpr": 0.0 }, "tpr60": 0.0 }
  },
  "aggregate": {
    "mean":       { "..." },
    "min":        { "..." },
    "best_layer": { "..." }
  }
}
```

Plots saved alongside `results.json`:
- `pr_curves.png` -- precision-recall curves per aggregation strategy
- `tpr_fnr_threshold.png` -- TPR/FNR vs threshold

## Directory structure

```
adv-glp/
├── glp/                          # Core GLP library (upstream)
│   ├── denoiser.py               # GLP model loading and denoising
│   ├── flow_matching.py          # Flow matching implementation
│   ├── script_eval.py            # Evaluation utilities
│   ├── script_probe.py           # 1-D scalar probing
│   ├── script_steer.py           # Steering utilities
│   └── utils_acts.py             # Activation extraction helpers
├── scripts/                      # Experiment scripts
│   ├── evaluate_classifier.py    # Guard-GLP classifiers (PI, DTE, recon error)
│   ├── evaluate_classifier.sh    # Slurm wrapper for classifier eval
│   ├── evaluate_all.sh           # Run all evaluations sequentially
│   ├── evaluate_diffmean.py      # DiffMean baseline
│   ├── evaluate_diffmean.sh
│   ├── evaluate_linear_probe.py  # Linear probe baseline
│   ├── evaluate_linear_probe.sh
│   ├── eval_linear_probe.sh      # Slurm wrapper for linear probe
│   ├── eval_diffmean.sh          # Slurm wrapper for DiffMean
│   ├── eval_ood.sh               # OOD evaluation
│   ├── steering.py               # Activation steering (benign-malicious)
│   ├── steering.sh               # Slurm wrapper
│   ├── steering_refusal.py       # Activation steering (refusal-compliance)
│   ├── steering_refusal.sh
│   ├── visualize_activations.py  # t-SNE / PCA activation plots
│   ├── visualize_activations.sh
│   ├── judge_responses.py        # LLM judge for steering responses
│   ├── rejudge_responses.py      # Re-judge existing response files
│   ├── rejudge.sh
│   ├── serve_llm.sh              # Serve vLLM judge model
│   ├── node1.sh                  # Batch: all steering variants (adversarial)
│   ├── node2.sh                  # Batch: all steering variants (refusal)
│   └── ood_*.py                  # OOD evaluation scripts
├── configs/
│   ├── paper/                    # Configs to reproduce paper results
│   │   ├── eval_plotting.yaml
│   │   ├── eval_pi.yaml          # Hutchinson path-integral
│   │   ├── eval_dte.yaml         # Diffusion time estimation
│   │   ├── eval_dte_glp.yaml     # DTE with GLP-sampled reference
│   │   ├── eval_reconstruction_err.yaml
│   │   ├── eval_lp.yaml          # Linear probe
│   │   ├── eval_diffmean.yaml    # DiffMean
│   │   └── steering/             # Steering experiment configs
│   ├── train_llama1b_static.yaml
│   └── train_llama8b_static.yaml
├── integrations/
│   └── persona_vectors/          # Persona Vectors integration (upstream)
├── environment.yaml              # Conda environment
├── pyproject.toml                # Package metadata and dependencies
└── requirements.txt
```

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@article{calanzone2025guardglp,
  title   = {Generative Meta-Models are Adversarial Classifiers},
  author  = {Diego Calanzone and Pietro Greiner},
  year    = {2025}
}
```

## Acknowledgements

This project builds on the [Generative Latent Prior (GLP)](https://arxiv.org/abs/2602.06964) codebase by Luo, Feng, Darrell, Radford, and Steinhardt. We thank the authors for releasing their pre-trained weights and code.