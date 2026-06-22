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
├── dataset/            # Build trainer-ready activation datasets
├── detection/          # Guard-GLP classifiers & baselines
├── steering/           # Activation steering experiments
├── inference/          # LLM judge & re-judging
└── visualization/      # t-SNE / PCA activation plots
```

See [`GUIDE.md`](../GUIDE.md) for the full, per-component run instructions.

## Dataset

Build activation datasets (extract per shard → finalize). One command submits the SLURM
pipeline: a GPU job array for extraction + a CPU-only finalize gated on its success.

```bash
bash scripts/dataset/build_activations.sh configs/dataset/build_wildchat_llama8b_layer24.yaml

# local / CPU (no SLURM): run the two passes directly
python scripts/dataset/build_activations.py run --config=configs/dataset/build_fineweb_llama1b_layer07.yaml --gpu_id=0
python scripts/dataset/build_activations.py finalize --config=configs/dataset/build_fineweb_llama1b_layer07.yaml
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
