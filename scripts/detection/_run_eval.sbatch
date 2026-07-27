#!/bin/bash
# Pass-1 worker: run the detection eval for one data shard.
# Submitted by scripts/detection/eval_classifier.sh as a job array (--array=0..N-1),
# one task per shard; each cgroup-isolates one GPU (visible as cuda:0). Can also be run
# directly to (re)compute a single shard, e.g.:
#   sbatch --array=3 --gpus-per-task=1 scripts/detection/_run_eval.sbatch CONFIG
#
# GPU isolation: GPUs are requested via --gpus-per-task (NOT --gres=gpu), because on
# this cluster only --gpus-per-task cgroup-isolates a *distinct* physical GPU per task;
# with --gres=gpu:1 every co-located task sees the same physical GPU 0 and they collide.
# Under the cgroup each task's GPU is renumbered to cuda:0, so --device=cuda:0 is correct
# while --gpu_id (the array index) stays the data-shard id.
#SBATCH -J glp-eval-shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_shard_%A_%a.out
#SBATCH --error=logs/eval_shard_%A_%a.err
set -uo pipefail

# Load shared environment (HF_HOME / UV_CACHE_DIR on the shared filesystem, HF_TOKEN).
set -a
[ -f .env ] && . ./.env
set +a

source .venv/bin/activate

CONFIG="$1"
SHARD=${SLURM_ARRAY_TASK_ID:-0}

python scripts/detection/evaluate_classifier.py run \
    --config="$CONFIG" --gpu_id="$SHARD" --device=cuda:0
