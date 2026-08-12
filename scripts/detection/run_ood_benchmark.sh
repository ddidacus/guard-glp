#!/bin/bash
# Full ID-vs-OOD detection benchmark (single-GPU). ID whitelist = guard-glp-benign
# WildChat; 6 OOD sets (3 jailbreak + 3 harmful). Compares training-free GLPs (new
# multi-layer + original) against supervised baselines (probe/mean-diff) in 3 regimes.
#
# Run from repo root on a GPU node (.venv active, .env sourced). Needs HF access to the
# gated sets (Llama-3.2-1B/-Instruct, wildjailbreak, AdvBench).
#   bash scripts/detection/run_ood_benchmark.sh
set -euo pipefail
set -a; [ -f .env ] && . ./.env; set +a

OOD="advbench harmbench harmbench_gcg wjb_vanilla wjb_adversarial toxicchat"

echo "############### GLP reconstruction-error (new + original), all 16 layers ########"
for glp in newglp origglp; do
    for o in $OOD; do
        cfg="configs/detection/ood/recon_${glp}_${o}.yaml"
        out="results/ood/recon-${glp}-${o}"
        echo ">>> $glp  ood:$o"
        python scripts/detection/evaluate_classifier.py run --config="$cfg" --gpu_id=0
        python scripts/detection/evaluate_classifier.py aggregate --out_dir="$out"
    done
done

echo "############### Supervised baselines (probe + mean-diff) x 3 regimes ############"
# one extraction pass (Instruct, mean-pooled, 16 layers) populates the shared cache,
# then aggregate trains all regimes and evaluates on every OOD test set.
python scripts/detection/eval_ood_baselines.py run --gpu_id=0
python scripts/detection/eval_ood_baselines.py aggregate --out_dir=results/ood/baselines

echo "############### Best-layer summary #############################################"
python - <<'PY'
import json
from pathlib import Path

OOD = ["advbench", "harmbench", "harmbench_gcg", "wjb_vanilla", "wjb_adversarial", "toxicchat"]


def best(path):
    try:
        bl = json.load(open(path))["aggregate"]["best_layer"]
        m = bl.get("recon_error", bl)
        return m["auprc"], m["auroc"], m.get("best_layer")
    except Exception:
        return None


print(f"\n=== GLP recon (best-layer AUPRC / AUROC) ===")
print(f"{'ood':<18}{'new-GLP':<22}{'orig-GLP':<22}")
for o in OOD:
    n = best(f"results/ood/recon-newglp-{o}/results.json")
    g = best(f"results/ood/recon-origglp-{o}/results.json")
    fn = f"{n[0]:.3f}/{n[1]:.3f} L{n[2]}" if n else "-"
    fg = f"{g[0]:.3f}/{g[1]:.3f} L{g[2]}" if g else "-"
    print(f"{o:<18}{fn:<22}{fg:<22}")

print(f"\n=== Supervised baselines (best-layer AUROC) — see results/ood/baselines/results.json ===")
try:
    r = json.load(open("results/ood/baselines/results.json"))["regimes"]
    for method, regimes in r.items():
        print(f"\n[{method}]")
        hdr = "  " + f"{'regime':<16}" + "".join(f"{o[:12]:<14}" for o in OOD)
        print(hdr)
        for regime, row in regimes.items():
            cells = "".join(f"{row[o]['auroc']:<14.3f}" for o in OOD)
            print(f"  {regime:<16}{cells}")
except Exception as e:
    print(f"  (no baseline results: {e})")
PY
