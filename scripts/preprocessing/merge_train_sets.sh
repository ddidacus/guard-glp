#!/bin/bash
#SBATCH -J glp-dataset-prep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:40gb:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --output=logs/merge_train_sets_%j.out
#SBATCH --error=logs/merge_train_sets_%j.err

export HF_HOME=$SCRATCH/.cache
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export UV_CACHE_DIR=$SCRATCH/.cache
export PYTHONPATH="$(pwd):$PYTHONPATH"

source .venv/bin/activate

# python scripts/preprocessing/merge_train_sets.py "$@"

python scripts/preprocessing/merge_train_sets.py \
    --output_dir data/guardglp_benign \
    --sim_theshold 0.99 \
    --semantic_dedup_samples_per_bucket 512 \
    --embed_batch_size 256 \
    --push_to_hub --private