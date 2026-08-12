#!/bin/bash
# Streaming-training stats pre-pass on a single GPU:
#   sbatch scripts/dataset/_stats.sbatch CONFIG [extra fire args...]
#
# Writes <output_dir>/rep_statistics.pt (+ stats_manifest.json) for a streaming
# training config to point its `rep_statistic` at. Same preflight requeue guard as
# scripts/train/_train.sbatch, so the job hops off a GPU held by an untracked process.
#SBATCH -J glp-stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --partition=defq
#SBATCH --time=6:00:00
#SBATCH --requeue
#SBATCH --output=logs/stats_%x_%j.out
#SBATCH --error=logs/stats_%x_%j.err
set -uo pipefail

set -a
[ -f .env ] && . ./.env
set +a
source .venv/bin/activate

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "${FREE_MIB:-0}" -lt 40000 ]; then
    echo "stats on $(hostname): GPU only ${FREE_MIB:-?} MiB free; requeue attempt=${ATTEMPT}"
    if [ "$ATTEMPT" -lt 15 ]; then
        scontrol requeue "$SLURM_JOB_ID"
        sleep 20
        exit 1
    fi
    echo "stats: no free GPU after ${ATTEMPT} requeues; giving up" >&2
    exit 1
fi

CONFIG="$1"
shift || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/dataset/compute_stats.py run --config="$CONFIG" "$@"
