#!/bin/bash
# Pass-1 worker: extract one data shard of activations.
# Submitted by scripts/dataset/build_activations.sh, which supplies the GPU request
# and (for hf_baukit) --array on the command line, overriding the defaults below.
# Can also be run directly to (re)build a single shard, e.g.:
#   sbatch --array=3 --gpus-per-task=1 scripts/dataset/_run_shard.sbatch CONFIG
#
# GPU isolation: we request GPUs via --gpus-per-task (NOT --gres=gpu), because on
# this cluster only --gpus-per-task cgroup-isolates a *distinct* physical GPU per
# task — with --gres=gpu:1 every co-located array task sees the same physical GPU 0
# and they collide (OOM / "free memory < utilization" at startup). Under the cgroup
# each task's assigned GPU is renumbered to cuda:0, so --device=cuda:0 is correct.
#SBATCH -J glp-build-shard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=defq
#SBATCH --time=8:00:00
#SBATCH --output=logs/build_shard_%A_%a.out
#SBATCH --error=logs/build_shard_%A_%a.err
set -uo pipefail

# Load shared environment (HF_HOME / UV_CACHE_DIR on the shared filesystem, HF_TOKEN).
set -a
[ -f .env ] && . ./.env
set +a

source .venv/bin/activate

CONFIG="$1"
SHARD=${SLURM_ARRAY_TASK_ID:-0}

# One GPU is cgroup-isolated by SLURM (visible as cuda:0); SHARD is only the
# data-shard index used to stride the corpus.
python scripts/dataset/build_activations.py run \
    --config="$CONFIG" --gpu_id="$SHARD" --device=cuda:0
