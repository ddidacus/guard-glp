#!/bin/bash
# Submit the two-pass detection eval to SLURM and exit:
#   pass 1 (GPU)  — data-parallel array of `num_gpus` tasks, each scores one shard
#                   (1 GPU per task, cgroup-isolated as cuda:0)
#   pass 2 (CPU)  — aggregate the per-shard outputs into results.json (+ plots)
# Pass 2 is submitted with --dependency=afterok on pass 1, so it only runs once every
# shard succeeds and never holds a GPU during the CPU-only merge.
#
# Run from the repo root (which must live on the shared filesystem):
#   bash scripts/detection/eval_classifier.sh [CONFIG]
#
# HF_HOME / UV_CACHE_DIR / HF_TOKEN come from .env (sourced inside the workers); point
# the caches at the shared filesystem there, not per-node $SCRATCH.
set -euo pipefail

CONFIG="${1:-configs/detection/eval_pi.yaml}"
mkdir -p logs

_cfg() { python -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print($1)"; }
NUM_GPUS=$(_cfg "c.get('num_gpus', 4)")
OUT_DIR=$(_cfg "c['out_dir']")

# Partition/constraint/exclude are cluster-specific; override per cluster without
# editing files (the #SBATCH defaults can't read env vars). CLI flags override #SBATCH.
PARTITION="${GLP_PARTITION:-defq}"
CONSTRAINT_FLAG=()
[ -n "${GLP_CONSTRAINT:-}" ] && CONSTRAINT_FLAG=(--constraint="${GLP_CONSTRAINT}")
EXCLUDE_FLAG=()
[ -n "${GLP_EXCLUDE:-}" ] && EXCLUDE_FLAG=(--exclude="${GLP_EXCLUDE}")

echo "config:    $CONFIG"
echo "out_dir:   $OUT_DIR   num_gpus(shards): $NUM_GPUS"
echo "partition: $PARTITION   constraint: ${GLP_CONSTRAINT:-<none>}   exclude: ${GLP_EXCLUDE:-<none>}"

# Pass 1: data-parallel GPU array. GPUs are requested with --gpus-per-task (NOT
# --gres=gpu) so each array task cgroup-isolates a distinct physical GPU (see
# _run_eval.sbatch).
JID=$(sbatch --parsable --partition="$PARTITION" "${CONSTRAINT_FLAG[@]}" "${EXCLUDE_FLAG[@]}" \
    --array=0-$((NUM_GPUS - 1)) --ntasks=1 --gpus-per-task=1 \
    scripts/detection/_run_eval.sbatch "$CONFIG")
echo "pass-1 (run) job: $JID"

# Pass 2: CPU-only aggregate, runs only if all pass-1 tasks succeed.
FID=$(sbatch --parsable --partition="$PARTITION" --dependency=afterok:"$JID" \
    scripts/detection/_aggregate_eval.sbatch "$OUT_DIR")
echo "pass-2 (aggregate) job: $FID   (afterok:$JID)"

echo "Submitted. Track with: squeue --me"
