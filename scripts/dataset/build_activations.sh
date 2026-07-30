#!/bin/bash
# Submit the two-pass dataset build to SLURM and exit:
#   pass 1 (GPU)  — extract activation shards
#   pass 2 (CPU)  — finalize: merge shards, write stats + manifest
# Pass 2 is submitted with --dependency=afterok on pass 1, so it only runs once
# every shard succeeds and never holds a GPU during the CPU-only merge.
#
# Run from the repo root (which must live on the shared filesystem):
#   bash scripts/dataset/build_activations.sh [CONFIG]
#
# Topology is chosen by `extract.tensor_parallel_size` (TP), not the backend:
#   TP == 1  -> data-parallel job array of `num_gpus` tasks, 1 GPU each (each task is
#               a full model on one GPU, striding the corpus). Used by hf_baukit and by
#               vllm_nnsight for models that fit on one GPU (the common case).
#   TP  > 1  -> a single task with TP GPUs (tensor parallel, one shard) for models too
#               large for one GPU. vllm_nnsight only.
set -euo pipefail

CONFIG="${1:-configs/dataset/build_wildchat_llama8b_layer24.yaml}"
mkdir -p logs

_cfg() { python -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print($1)"; }
BACKEND=$(_cfg "c.get('backend', 'hf_baukit')")
NUM_GPUS=$(_cfg "c.get('num_gpus', 1)")
TP=$(_cfg "c.get('extract', {}).get('tensor_parallel_size', 1)")

# Partition/constraint/exclude are cluster-specific; override per cluster without
# editing files (the #SBATCH defaults can't read env vars). CLI flags override #SBATCH.
#   GLP_EXCLUDE: comma-separated nodes to avoid, e.g. nodes with untracked orphan
#   processes holding GPU memory that SLURM still reports as idle (a shard scheduled
#   there aborts at startup with "Free memory on device ... less than utilization").
PARTITION="${GLP_PARTITION:-defq}"
CONSTRAINT_FLAG=()
[ -n "${GLP_CONSTRAINT:-}" ] && CONSTRAINT_FLAG=(--constraint="${GLP_CONSTRAINT}")
EXCLUDE_FLAG=()
[ -n "${GLP_EXCLUDE:-}" ] && EXCLUDE_FLAG=(--exclude="${GLP_EXCLUDE}")

echo "config:  $CONFIG"
echo "backend: $BACKEND   num_gpus(shards): $NUM_GPUS   tensor_parallel_size: $TP"
echo "partition: $PARTITION   constraint: ${GLP_CONSTRAINT:-<none>}   exclude: ${GLP_EXCLUDE:-<none>}"

# Pass 1: GPU extraction. GPUs are requested with --gpus-per-task (NOT --gres=gpu):
# only --gpus-per-task cgroup-isolates a distinct physical GPU per task on this
# cluster, so co-located array shards don't collide on GPU 0 (see _run_shard.sbatch).
if [ "$TP" -gt 1 ]; then
    # Tensor parallel: one shard split across TP GPUs (model too large for one GPU).
    JID=$(sbatch --parsable --partition="$PARTITION" "${CONSTRAINT_FLAG[@]}" "${EXCLUDE_FLAG[@]}" \
        --ntasks=1 --gpus-per-task="$TP" scripts/dataset/_run_shard.sbatch "$CONFIG")
else
    # Data parallel: `num_gpus` independent shards, one full model per GPU.
    JID=$(sbatch --parsable --partition="$PARTITION" "${CONSTRAINT_FLAG[@]}" "${EXCLUDE_FLAG[@]}" \
        --array=0-$((NUM_GPUS - 1)) --ntasks=1 --gpus-per-task=1 \
        scripts/dataset/_run_shard.sbatch "$CONFIG")
fi
echo "pass-1 (extract) job: $JID"

# Pass 2: CPU-only finalize, runs only if all pass-1 tasks succeed.
FID=$(sbatch --parsable --partition="$PARTITION" --dependency=afterok:"$JID" \
    scripts/dataset/_finalize.sbatch "$CONFIG")
echo "pass-2 (finalize) job: $FID   (afterok:$JID)"

echo "Submitted. Track with: squeue --me"
