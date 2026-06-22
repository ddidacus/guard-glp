#!/bin/bash
# Pass-2 worker: merge all shards into final per-(granularity, layer) datasets and
# write dtype.txt / rep_statistics.pt / manifest.json. CPU-only (no GPU), submitted
# by scripts/dataset/build_activations.sh with --dependency=afterok on pass 1.
#SBATCH -J glp-build-finalize
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=2:00:00
#SBATCH --output=logs/build_finalize_%j.out
#SBATCH --error=logs/build_finalize_%j.err
set -uo pipefail

set -a
[ -f .env ] && . ./.env
set +a

source .venv/bin/activate

CONFIG="$1"
python scripts/dataset/build_activations.py finalize --config="$CONFIG"
