#!/bin/bash
# Launch the full guard-glp-benign activation campaign: BOTH prompt views
# (useronly + full), each extracting layers [8, 12, 14] last-token in one pass.
# The 6 output datasets (2 views x 3 layers) are produced by these 2 runs.
#
#   bash scripts/dataset/run_guardglpbenign_campaign.sh
#
# Each config is submitted through the standard two-pass SLURM pipeline
# (scripts/dataset/build_activations.sh): a GPU job array of `num_gpus` shards +
# a CPU-only `finalize` that runs via --dependency=afterok. The two views are
# independent job chains and run concurrently. Submission only needs python+pyyaml;
# the sbatch workers activate .venv and source .env themselves. Partition/constraint
# overrides (GLP_PARTITION / GLP_CONSTRAINT) are passed through to build_activations.sh.
#
# Outputs: data/llama1b-guardglpbenign-{useronly,full}/last/layer_{08,12,14}/
# Track with: squeue --me   (logs in logs/build_{shard,finalize}_*.{out,err})
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"

CONFIGS=(
    configs/dataset/build_guardglpbenign_llama1b_last_useronly.yaml
    configs/dataset/build_guardglpbenign_llama1b_last_full.yaml
)

for CFG in "${CONFIGS[@]}"; do
    echo "=============================================================="
    echo "### submitting campaign build: $CFG"
    echo "=============================================================="
    bash scripts/dataset/build_activations.sh "$CFG"
    echo
done

echo "All campaign jobs submitted. Track with: squeue --me"
