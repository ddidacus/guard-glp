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
# Backend-aware (read from the config):
#   hf_baukit     -> job array of `num_gpus` tasks, 1 GPU each (data-parallel shards)
#   vllm_nnsight  -> a single task with `extract.tensor_parallel_size` GPUs (1 shard)
set -euo pipefail

CONFIG="${1:-configs/dataset/build_wildchat_llama8b_layer24.yaml}"
mkdir -p logs

_cfg() { python -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print($1)"; }
BACKEND=$(_cfg "c.get('backend', 'hf_baukit')")
NUM_GPUS=$(_cfg "c.get('num_gpus', 1)")
TP=$(_cfg "c.get('extract', {}).get('tensor_parallel_size', 1)")

echo "config:  $CONFIG"
echo "backend: $BACKEND   num_gpus(shards): $NUM_GPUS   tensor_parallel_size: $TP"

# Pass 1: GPU extraction. CLI flags (--array/--gres) override the worker's #SBATCH.
if [ "$BACKEND" = "vllm_nnsight" ]; then
    JID=$(sbatch --parsable --gres=gpu:"$TP" \
        scripts/dataset/_run_shard.sbatch "$CONFIG")
else
    JID=$(sbatch --parsable --array=0-$((NUM_GPUS - 1)) --gres=gpu:1 \
        scripts/dataset/_run_shard.sbatch "$CONFIG")
fi
echo "pass-1 (extract) job: $JID"

# Pass 2: CPU-only finalize, runs only if all pass-1 tasks succeed.
FID=$(sbatch --parsable --dependency=afterok:"$JID" \
    scripts/dataset/_finalize.sbatch "$CONFIG")
echo "pass-2 (finalize) job: $FID   (afterok:$JID)"

echo "Submitted. Track with: squeue --me"
