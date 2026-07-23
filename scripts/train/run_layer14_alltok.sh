#!/bin/bash
# Launch the two layer-14 all-token GLP trainings in parallel (useronly + full),
# one GPU each, via scripts/train/_train.sbatch.
#   bash scripts/train/run_layer14_alltok.sh
# Override the excluded nodes with GLP_EXCLUDE (e.g. GLP_EXCLUDE=dgx-43,dgx-45 ...).
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")"

CONFIG="configs/train/glp_llama1b_guardglpbenign_alltok_layer14.yaml"
PARTITION="${GLP_PARTITION:-defq}"
EXCLUDE_FLAG=()
[ -n "${GLP_EXCLUDE:-}" ] && EXCLUDE_FLAG=(--exclude="${GLP_EXCLUDE}")

for VIEW in useronly full; do
    echo "### submitting GLP training: view=${VIEW} layer=14"
    sbatch --partition="$PARTITION" "${EXCLUDE_FLAG[@]}" \
        --job-name="glp-train-${VIEW}-l14" \
        scripts/train/_train.sbatch "$CONFIG" "$VIEW" 14
done
echo "Submitted. Track with: squeue --me   (wandb: project guard-glp)"
