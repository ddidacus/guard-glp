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

# Partition/constraint are cluster-specific; override per cluster without editing
# files (the #SBATCH defaults can't read env vars). CLI flags override #SBATCH.
PARTITION="${GLP_PARTITION:-defq}"
CONSTRAINT_FLAG=()
[ -n "${GLP_CONSTRAINT:-}" ] && CONSTRAINT_FLAG=(--constraint="${GLP_CONSTRAINT}")

echo "config:  $CONFIG"
echo "backend: $BACKEND   num_gpus(shards): $NUM_GPUS   tensor_parallel_size: $TP"
echo "partition: $PARTITION   constraint: ${GLP_CONSTRAINT:-<none>}"

# Pass 1: GPU extraction. CLI flags (--array/--gres) override the worker's #SBATCH.
if [ "$TP" -gt 1 ]; then
    # Tensor parallel: one shard split across TP GPUs (model too large for one GPU).
    JID=$(sbatch --parsable --partition="$PARTITION" "${CONSTRAINT_FLAG[@]}" \
        --gres=gpu:"$TP" scripts/dataset/_run_shard.sbatch "$CONFIG")
else
    # Data parallel: `num_gpus` independent shards, one full model per GPU.
    JID=$(sbatch --parsable --partition="$PARTITION" "${CONSTRAINT_FLAG[@]}" \
        --array=0-$((NUM_GPUS - 1)) --gres=gpu:1 \
        scripts/dataset/_run_shard.sbatch "$CONFIG")
fi
echo "pass-1 (extract) job: $JID"

# Pass 2: CPU-only finalize, runs only if all pass-1 tasks succeed.
FID=$(sbatch --parsable --partition="$PARTITION" --dependency=afterok:"$JID" \
    scripts/dataset/_finalize.sbatch "$CONFIG")
echo "pass-2 (finalize) job: $FID   (afterok:$JID)"

echo "Submitted. Track with: squeue --me"
