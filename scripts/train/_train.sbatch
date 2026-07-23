#!/bin/bash
# Train one GLP on a single GPU. Submitted by scripts/train/run_layer14_alltok.sh, or
# directly:  sbatch scripts/train/_train.sbatch CONFIG VIEW [LAYER]
#
# Uses --gpus-per-task (cgroup-isolates a distinct GPU) and the same preflight
# requeue guard as the extraction worker, so a job placed on a GPU occupied by an
# untracked process hops to a free one instead of OOM-ing.
#SBATCH -J glp-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --partition=defq
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err
set -uo pipefail

set -a
[ -f .env ] && . ./.env
set +a
source .venv/bin/activate

# Preflight: skip GPUs occupied by untracked processes (see scripts/dataset/_run_shard.sbatch).
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
ATTEMPT=${SLURM_RESTART_COUNT:-0}
if [ "${FREE_MIB:-0}" -lt 40000 ]; then
    echo "train on $(hostname): GPU only ${FREE_MIB:-?} MiB free; requeue attempt=${ATTEMPT}"
    if [ "$ATTEMPT" -lt 15 ]; then
        scontrol requeue "$SLURM_JOB_ID"
        sleep 20
        exit 1
    fi
    echo "train: no free GPU after ${ATTEMPT} requeues; giving up" >&2
    exit 1
fi

CONFIG="$1"
VIEW="$2"
LAYER="${3:-14}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/train/train_glp.py config="$CONFIG" view="$VIEW" layer="$LAYER" device=cuda:0
