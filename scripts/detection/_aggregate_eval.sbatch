#!/bin/bash
# Pass-2 worker: aggregate the per-shard eval outputs into results.json (+ PR/threshold
# plots). CPU-only; submitted by scripts/detection/eval_classifier.sh with
# --dependency=afterok on the pass-1 array, so it runs only once every shard succeeds.
#SBATCH -J glp-eval-aggregate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=defq
#SBATCH --time=1:00:00
#SBATCH --output=logs/eval_aggregate_%j.out
#SBATCH --error=logs/eval_aggregate_%j.err
set -uo pipefail

set -a
[ -f .env ] && . ./.env
set +a

source .venv/bin/activate

OUT_DIR="$1"

python scripts/detection/evaluate_classifier.py aggregate --out_dir="$OUT_DIR"
