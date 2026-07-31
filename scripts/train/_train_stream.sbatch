#!/bin/bash
# Streaming GLP training on one full node: extraction producers + DDP trainers.
#   sbatch scripts/train/_train_stream.sbatch CONFIG [extra overrides...]
#
# The launcher (train_glp_stream.py) partitions the node's GPUs itself: the
# first streaming.num_producers become extraction producers, the rest DDP
# trainer ranks. Same preflight requeue guard as _train.sbatch, applied to all
# GPUs (a whole-node job needs the whole node free).
#
# NB: with streaming.backend=vllm_nnsight, `.venv` must include the `serve` extra
# (`uv sync --extra serve`), same as the dataset workers. The lockfile pins
# transformers==4.47.0 via a [tool.uv] override (vLLM 0.9.2 can't import under
# transformers>=4.49), so there is no manual version step. See GUIDE.md.
#SBATCH -J glp-train-stream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/train_stream_%x_%j.out
#SBATCH --error=logs/train_stream_%x_%j.err
set -uo pipefail

set -a
[ -f .env ] && . ./.env
set +a
source .venv/bin/activate

# Preflight: every GPU must be free of untracked processes (see _train.sbatch).
MIN_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "${MIN_FREE:-0}" -lt 40000 ]; then
    echo "train-stream on $(hostname): a GPU has only ${MIN_FREE:-?} MiB free; requeue attempt=${ATTEMPT}"
    if [ "$ATTEMPT" -lt 15 ]; then
        scontrol requeue "$SLURM_JOB_ID"
        sleep 20
        exit 1
    fi
    echo "train-stream: no free node after ${ATTEMPT} requeues; giving up" >&2
    exit 1
fi

CONFIG="$1"
shift || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train/train_glp_stream.py config="$CONFIG" "$@"
