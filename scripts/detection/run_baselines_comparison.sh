#!/bin/bash
# Run the supervised/statistical detection baselines — linear probe and
# difference-of-means — on both datasets (guard_glp_data, wildjailbreak_vanilla),
# single-GPU, at layer 14, and print a best-layer AUPRC comparison table. Companion to
# run_recon_err_comparison.sh (the GLP reconstruction-error runs).
#
# Run from the repo root on a GPU node (activate .venv first). Requires HF_TOKEN with
# accepted licenses for meta-llama/Llama-3.2-1B-Instruct and allenai/wildjailbreak.
#   bash scripts/detection/run_baselines_comparison.sh [--fresh]
#
#   --fresh   delete each out_dir first (drop cached activations / stale shard files)
set -euo pipefail

set -a
[ -f .env ] && . ./.env
set +a

FRESH=0
[ "${1:-}" = "--fresh" ] && FRESH=1

# script : config-basename : out_dir
JOBS=(
    "eval_linear_probe:eval_lp_guardglpbenign:results/eval-lp-guardglpbenign"
    "eval_linear_probe:eval_lp_wildjailbreak:results/eval-lp-wildjailbreak"
    "eval_diffmean:eval_diffmean_guardglpbenign:results/eval-diffmean-guardglpbenign"
    "eval_diffmean:eval_diffmean_wildjailbreak:results/eval-diffmean-wildjailbreak"
)

for entry in "${JOBS[@]}"; do
    IFS=":" read -r script name out_dir <<<"$entry"
    config="configs/detection/${name}.yaml"
    echo "================================================================"
    echo ">>> $script  $name"
    echo "================================================================"
    if [ "$FRESH" -eq 1 ]; then
        echo "[fresh] removing $out_dir"
        rm -rf "$out_dir"
    fi
    python "scripts/detection/${script}.py" run --config="$config" --gpu_id=0
    python "scripts/detection/${script}.py" aggregate --out_dir="$out_dir"
done

echo
echo "================================================================"
echo " Baselines (best layer) — comparison"
echo "================================================================"
printf "  %-38s %8s %8s\n" "config" "AUPRC" "AUROC"
for entry in "${JOBS[@]}"; do
    IFS=":" read -r _script name out_dir <<<"$entry"
    python - "$name" "$out_dir/results.json" <<'PY'
import json
import sys

name, path = sys.argv[1], sys.argv[2]


def _metric(bl, key):
    # probe/diffmean store metrics flat on best_layer; recon nests under a metric key.
    v = bl.get(key)
    if v is None:
        v = next(
            (d[key] for d in bl.values() if isinstance(d, dict) and key in d), None
        )
    return v


try:
    with open(path) as f:
        r = json.load(f)
    bl = r["aggregate"]["best_layer"]
    auprc, auroc = _metric(bl, "auprc"), _metric(bl, "auroc")
    print(f"  {name:<38} {auprc:8.4f} {auroc:8.4f}")
except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
    print(f"  {name:<38} (no result: {e})")
PY
done
