# scripts/

All scripts run **from the project root** with the virtual environment activated.

```bash
export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
source .venv/bin/activate
```

## Layout

```
scripts/
├── detection/          # Guard-GLP classifiers & baselines
├── steering/           # Activation steering experiments
├── inference/          # LLM judge & re-judging
└── visualization/      # t-SNE / PCA activation plots
```

## Detection

Two-pass workflow: (1) parallel per-GPU extraction, (2) single-process aggregation.

```bash
# Individual classifiers
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_pi.yaml
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_dte.yaml
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_dte_glp.yaml
sbatch scripts/detection/eval_classifier.sh configs/detection/eval_reconstruction_err.yaml
sbatch scripts/detection/eval_linear_probe.sh configs/detection/eval_lp.yaml
sbatch scripts/detection/eval_diffmean.sh configs/detection/eval_diffmean.yaml

# All at once
bash scripts/detection/eval_all.sh

# OOD evaluation
sbatch scripts/detection/eval_ood.sh
```

## Steering

```bash
bash scripts/steering/run_steering_benign.sh
bash scripts/steering/run_steering_refusal.sh
```

Configs: `configs/paper/steering/`

## Inference (judge / re-judge)

```bash
sbatch scripts/inference/serve_llm.sh
RESULTS_DIR=results/steering_glp bash scripts/inference/rejudge.sh
bash scripts/inference/rejudge_all.sh
```

## Visualisation

```bash
sbatch scripts/visualization/visualize_activations.sh
```

Config: `configs/visualization/eval_plotting.yaml`
