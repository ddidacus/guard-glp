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
#SBATCH --requeue
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

# Preflight: SLURM's GRES accounting only tracks *allocation*, not actual GPU memory.
# On this cluster some nodes SLURM reports as idle still carry untracked processes
# (another job's leaked vLLM workers) holding ~all of a GPU's memory, so a shard
# scheduled there aborts at startup ("Free memory on device ... less than utilization").
# Rather than fail the whole array, check the cgroup-assigned GPU (cuda:0) and, if it
# is not actually free, requeue this task so the scheduler re-places it elsewhere.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "${FREE_MIB:-0}" -lt 40000 ]; then
    echo "shard $SHARD on $(hostname): assigned GPU has only ${FREE_MIB:-?} MiB free" \
         "(occupied by an untracked process); requeue attempt=${ATTEMPT}"
    if [ "$ATTEMPT" -lt 15 ]; then
        scontrol requeue "$SLURM_JOB_ID"
        sleep 20
        exit 1   # SLURM re-runs this task (requeued) on a fresh placement
    fi
    echo "shard $SHARD: no free GPU after ${ATTEMPT} requeues; giving up" >&2
    exit 1
fi
echo "shard $SHARD on $(hostname): GPU OK (${FREE_MIB} MiB free)"

# Reduce CUDA allocator fragmentation over a long-running shard (the nnsight capture
# path churns large activation tensors; fragmentation otherwise strands GBs and OOMs).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# One GPU is cgroup-isolated by SLURM (visible as cuda:0); SHARD is only the
# data-shard index used to stride the corpus.
python scripts/dataset/build_activations.py run \
    --config="$CONFIG" --gpu_id="$SHARD" --device=cuda:0
