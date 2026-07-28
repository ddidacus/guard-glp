#!/bin/bash
# Run the reconstruction-error detection eval for the two new layer-14 GLPs (full,
# useronly) on both datasets (guard_glp_data, wildjailbreak_vanilla), single-GPU, and
# print an AUPRC comparison table. Each config is num_gpus=1, so one --gpu_id=0 run
# scores the whole eval set (~1 min each on an H100).
#
# Run from the repo root on a GPU node (activate .venv first). Requires HF_TOKEN with
# accepted licenses for meta-llama/Llama-3.2-1B-Instruct and allenai/wildjailbreak.
#   bash scripts/detection/run_recon_err_comparison.sh [--fresh]
#
#   --fresh   delete each out_dir first (drop cached activations / stale shard files)
set -euo pipefail

# Load shared environment (HF_HOME / UV_CACHE_DIR / HF_TOKEN) if a manual shell hasn't.
set -a
[ -f .env ] && . ./.env
set +a

FRESH=0
[ "${1:-}" = "--fresh" ] && FRESH=1

# config basename -> out_dir (out_dir also read from the yaml, but we need it up front
# for --fresh and the summary; keep them in lockstep with the configs).
CONFIGS=(
    "eval_recon_err_offtheshelf_guardglpbenign:results/eval-recon_err-offtheshelf-guardglpbenign"
    "eval_recon_err_offtheshelf_wildjailbreak:results/eval-recon_err-offtheshelf-wildjailbreak"
    "eval_recon_err_guardglpbenign_full:results/eval-recon_err-guardglpbenign-full"
    "eval_recon_err_guardglpbenign_useronly:results/eval-recon_err-guardglpbenign-useronly"
    "eval_recon_err_wildjailbreak_full:results/eval-recon_err-wildjailbreak-full"
    "eval_recon_err_wildjailbreak_useronly:results/eval-recon_err-wildjailbreak-useronly"
)

for entry in "${CONFIGS[@]}"; do
    name="${entry%%:*}"
    out_dir="${entry##*:}"
    config="configs/detection/${name}.yaml"
    echo "================================================================"
    echo ">>> $name"
    echo "================================================================"
    if [ "$FRESH" -eq 1 ]; then
        echo "[fresh] removing $out_dir"
        rm -rf "$out_dir"
    fi
    python scripts/detection/evaluate_classifier.py run --config="$config" --gpu_id=0
    python scripts/detection/evaluate_classifier.py aggregate --out_dir="$out_dir"
done

# Summary: pull the best-layer AUPRC out of each results.json.
echo
echo "================================================================"
echo " Reconstruction-error (best layer) — comparison"
echo "================================================================"
printf "  %-42s %8s %8s\n" "config" "AUPRC" "AUROC"
for entry in "${CONFIGS[@]}"; do
    name="${entry%%:*}"
    out_dir="${entry##*:}"
    python - "$name" "$out_dir/results.json" <<'PY'
import json
import sys

name, path = sys.argv[1], sys.argv[2]


def _metric(bl, key):
    # recon nests metrics under a metric key (recon_error); flat fallback otherwise.
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
    print(f"  {name:<42} {auprc:8.4f} {auroc:8.4f}")
except (OSError, KeyError, TypeError, json.JSONDecodeError) as e:
    print(f"  {name:<42} (no result: {e})")
PY
done
