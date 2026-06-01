# Guard-GLP: Generative Meta-Models are Adversarial Classifiers

[![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg?style=for-the-badge)](https://arxiv.org/)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Dataset on HF](https://img.shields.io/badge/Dataset-HuggingFace-FDEE21?style=for-the-badge&logo=HuggingFace&logoColor=black)](https://huggingface.co/datasets/ddidacus/guard-glp-data)

**Authors**: Diego Calanzone, Pietro Greiner. <br>
**Affiliation**: Mila Quebec AI Institute, Universite de Montreal.

We study adversarial prompt detection in LLMs through the geometry of intermediate activations. Our hypothesis is that token-level adversarial attacks induce hidden states that lie outside the activation manifold associated with typical prompts. To test this, we use Generative Latent Priors (GLP), diffusion-based meta-models trained on LLM residual stream activations, as proxies for this manifold. We evaluate several training-free anomaly scores derived from GLP, namely **Guard-GLP**, including reconstruction error, diffusion time estimation, and a density-based estimate using Hutchinson trace estimation. Empirically, GLP reconstruction error separates benign prompts from adversarial prompts and provides a competitive classifier without supervised fine-tuning. We also study GLP as a regularizer for activation steering, showing that denoising edited activations can reduce attack success while limiting over-refusal and nonsensical generations. Overall, our results suggest that generative priors over LLM activations provide a useful interface for both adversarial prompt detection and safer activation-level interventions.

## Setup

Tested with Python 3.12. Builds on the [GLP codebase](https://github.com/generative-latent-prior/glp) by Luo et al., 2026.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install vllm==0.9.2
uv pip install transformers==4.47.0
uv pip install -e .
```
> **Note:** Install in this exact order and ignore pip warnings. This is the only combination that makes vllm/nnsight/transformers work together.

Set cache paths (required on the Mila cluster):
```bash
export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
```

## Pre-Trained Weights

We use pre-trained GLP checkpoints from [generative-latent-prior](https://huggingface.co/generative-latent-prior):

```python
from glp.denoiser import load_glp
model = load_glp("generative-latent-prior/glp-llama1b-d6", device="cuda:0", checkpoint="final")
```

| Model | HuggingFace |
|-|-|
| GLP Llama-1B (d=6) | [Link](https://huggingface.co/generative-latent-prior/glp-llama1b-d6) |
| GLP Llama-8B (d=6) | [Link](https://huggingface.co/generative-latent-prior/glp-llama8b-d6) |

## Dataset

The dataset [`ddidacus/guard-glp-data`](https://huggingface.co/datasets/ddidacus/guard-glp-data) is downloaded automatically by all evaluation scripts.

| Split | Size | Benign | Malicious |
|-------|-----:|-------:|----------:|
| train | ~7 700 | ~3 500 | ~4 200 |
| calibration | ~1 100 | ~500 | ~600 |
| test | ~2 200 | ~1 000 | ~1 200 |

Fields: `prompt` (string), `adversarial` (bool, `True` = malicious).

To rebuild from upstream sources:
```bash
python scripts/create_dataset.py
```

---

## Reproducing the Experiments

All scripts run **from the project root** with the virtual environment activated.

### 1. Activation visualisation (t-SNE / PCA)

```bash
sbatch scripts/visualization/visualize_activations.sh
```

Config: `configs/visualization/eval_plotting.yaml`

### 2. Adversarial prompt detection (Guard-GLP)

All classifiers use a two-pass workflow: (1) parallel per-GPU extraction, (2) aggregation into `results.json`.

```bash
# GLP reconstruction error
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_reconstruction_err.yaml

# GLP path-integral density (Hutchinson)
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_pi.yaml

# Diffusion time estimation (DTE)
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_dte.yaml

# GLP-DTE (GLP-sampled reference)
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_dte_glp.yaml

# Linear probe (supervised baseline)
sbatch scripts/detection/eval_linear_probe.sh configs/detection/eval_lp.yaml

# DiffMean (baseline)
sbatch scripts/detection/eval_diffmean.sh configs/detection/eval_diffmean.yaml

# Run all at once
bash scripts/detection/eval_all.sh
```

### 3. OOD evaluation

```bash
sbatch scripts/detection/eval_ood.sh
```

### 4. Activation steering

GLP-regularised activation steering on adversarial prompts, judged for safety with Llama-Guard-3-8B.

```bash
# Benign-malicious steering (all variants: none, sv, glp)
bash scripts/steering/run_steering_benign.sh

# Refusal-compliance steering
bash scripts/steering/run_steering_refusal.sh
```

Steering configs live in `configs/paper/steering/`.

### 5. Re-judging responses

Re-judge previously generated responses with an LLM judge served via vLLM:

```bash
# Terminal 1: serve the judge
sbatch scripts/inference/serve_llm.sh

# Terminal 2: re-judge a results directory
RESULTS_DIR=results/steering_glp bash scripts/inference/rejudge.sh

# Re-judge all steering results
bash scripts/inference/rejudge_all.sh
```

---

## Output format

Every classifier writes a `results.json` to its output directory:

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

Plots saved alongside: `pr_curves.png`, `tpr_fnr_threshold.png`.

## Citation

```bibtex
@article{calanzone2025guardglp,
  title   = {Generative Meta-Models are Adversarial Classifiers},
  author  = {Diego Calanzone and Pietro Greiner},
  year    = {2025}
}
```

## Acknowledgements

This project builds on the [Generative Latent Prior (GLP)](https://arxiv.org/abs/2602.06964) codebase by Luo, Feng, Darrell, Radford, and Steinhardt. We thank the authors for releasing their pre-trained weights and code.
