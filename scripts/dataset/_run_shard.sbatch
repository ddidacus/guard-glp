#!/bin/bash
# Pass-1 worker: extract one data shard of activations.
# Submitted by scripts/dataset/build_activations.sh, which supplies --gres and
# (for hf_baukit) --array on the command line, overriding the defaults below.
# Can also be run directly to (re)build a single shard, e.g.:
#   sbatch --array=3 --gres=gpu:1 scripts/dataset/_run_shard.sbatch CONFIG
#SBATCH -J glp-build-shard
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --partition=long
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

# One GPU is allocated and pinned by SLURM (visible as cuda:0); SHARD is only the
# data-shard index used to stride the corpus.
python scripts/dataset/build_activations.py run \
    --config="$CONFIG" --gpu_id="$SHARD" --device=cuda:0
