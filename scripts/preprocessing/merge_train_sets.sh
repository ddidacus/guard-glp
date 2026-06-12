#!/bin/bash
#SBATCH -J glp-dataset-prep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --gres=gpu:a100l:1
#SBATCH --partition=long
#SBATCH --time=72:00:00
#SBATCH --output=logs/merge_train_sets_%j.out
#SBATCH --error=logs/merge_train_sets_%j.err

export HF_HOME=$SCRATCH/.cache
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

python scripts/preprocessing/merge_train_sets.py "$@"
