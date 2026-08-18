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

# The chunk queues pass tensors by fd (see _configure_shm in train_glp_stream.py), so
# give the job all the descriptors the node allows.
ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

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

# scancel must not look like a crash to the resubmit logic below.
CANCELLED=0
trap 'CANCELLED=1' TERM

python scripts/train/train_glp_stream.py config="$CONFIG" "$@"
RC=$?
[ "$RC" -eq 0 ] && exit 0
[ "$CANCELLED" -eq 1 ] && { echo "train-stream: cancelled; not resubmitting"; exit "$RC"; }

# An application failure is NOT covered by --requeue (that is preemption/node failure
# only), so a multi-day run would otherwise just end and leave the node idle. Resubmit
# ourselves and let the config's preset resume_from continue from the last checkpoint.
# Bounded, and only when a checkpoint exists, so a config error cannot loop.
ATTEMPT=${GLP_RESUBMIT_ATTEMPT:-0}
MAX=${GLP_RESUBMIT_MAX:-10}
RUN_DIR=$(python -c '
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.merge(
    OmegaConf.load(sys.argv[1]),
    OmegaConf.from_dotlist([a for a in sys.argv[2:] if "=" in a]),
)
print(OmegaConf.to_container(cfg, resolve=True)["output_path"])
' "$CONFIG" "$@" 2>/dev/null)

if [ "$ATTEMPT" -lt "$MAX" ] && [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/train_state.pt" ]; then
    TIMELIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o "%l" | tr -d ' ')
    echo "train-stream: exit $RC; resubmitting to resume from $RUN_DIR" \
         "(attempt $((ATTEMPT + 1))/$MAX, time=${TIMELIMIT:-24:00:00})"
    sbatch --time="${TIMELIMIT:-24:00:00}" \
        --export="ALL,GLP_RESUBMIT_ATTEMPT=$((ATTEMPT + 1)),GLP_RESUBMIT_MAX=$MAX" \
        scripts/train/_train_stream.sbatch "$CONFIG" "$@"
else
    echo "train-stream: exit $RC; NOT resubmitting" \
         "(attempt=$ATTEMPT/$MAX, checkpoint=${RUN_DIR:-?}/train_state.pt)" >&2
fi
exit "$RC"
