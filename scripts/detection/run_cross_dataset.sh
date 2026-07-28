#!/bin/bash
# Cross-dataset generalization for the supervised baselines: train on one dataset,
# test on the other, for both linear-probe and diff-of-means, in both OOD directions.
# Tests whether the near-perfect in-distribution scores transfer or are overfitting.
#
# PASS-2 ONLY: reuses the shared activation cache (results/_activation_cache), so run
# the baseline extraction first to populate it:
#     bash scripts/detection/run_baselines_comparison.sh
# then:
#     bash scripts/detection/run_cross_dataset.sh
#
# CPU-capable (no GPU needed — it only trains a linear probe / computes a mean vector
# on cached activations).
set -euo pipefail

set -a
[ -f .env ] && . ./.env
set +a

GG=guard_glp_data
WJB=wildjailbreak_vanilla

# method : train : test : out_dir
JOBS=(
    "probe:$GG:$WJB:results/xeval-probe-gg2wjb"
    "probe:$WJB:$GG:results/xeval-probe-wjb2gg"
    "diffmean:$GG:$WJB:results/xeval-diffmean-gg2wjb"
    "diffmean:$WJB:$GG:results/xeval-diffmean-wjb2gg"
)

for entry in "${JOBS[@]}"; do
    IFS=":" read -r method train test out_dir <<<"$entry"
    echo "================================================================"
    echo ">>> $method   train=$train  ->  test=$test"
    echo "================================================================"
    python scripts/detection/eval_cross_dataset.py \
        --method="$method" --train_dataset="$train" --test_dataset="$test" \
        --out_dir="$out_dir"
done

echo
echo "================================================================"
echo " Cross-dataset generalization (best layer) — OOD"
echo "================================================================"
printf "  %-10s %-22s %-22s %8s %8s\n" "method" "train" "test" "AUPRC" "AUROC"
for entry in "${JOBS[@]}"; do
    IFS=":" read -r method train test out_dir <<<"$entry"
    python - "$method" "$train" "$test" "$out_dir/results.json" <<'PY'
import json
import sys

method, train, test, path = sys.argv[1:5]
try:
    with open(path) as f:
        bl = json.load(f)["aggregate"]["best_layer"]
    print(f"  {method:<10} {train:<22} {test:<22} {bl['auprc']:8.4f} {bl['auroc']:8.4f}")
except (OSError, KeyError, json.JSONDecodeError) as e:
    print(f"  {method:<10} {train:<22} {test:<22} (no result: {e})")
PY
done
