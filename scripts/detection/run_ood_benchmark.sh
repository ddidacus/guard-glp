#!/bin/bash
# Full ID-vs-OOD detection benchmark, parallelized across all visible GPUs.
# ID whitelist = guard-glp-benign WildChat; OOD = wjb_vanilla, wjb_adversarial (jailbreak)
# + advbench, harmbench, toxicchat (harmful). Compares training-free GLPs (new
# multi-layer + original) against supervised baselines (probe/mean-diff) in 3 regimes.
#
# Strategy: the ID activations are shared across all tasks (keyed 'id_pool'), so we WARM
# the ID cache serially once per LLM (avoids cold-cache races), then fan the remaining
# independent recon configs + the baseline extraction out across GPUs, then aggregate
# (CPU) and print summary tables.
#
# Run from repo root on a GPU node (.venv active, .env sourced).
#   bash scripts/detection/run_ood_benchmark.sh
set -euo pipefail
set -a; [ -f .env ] && . ./.env; set +a

OOD=(advbench harmbench wjb_vanilla wjb_adversarial toxicchat)
GLPS=(newglp origglp)

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[+] Using $NGPU GPU(s)"
mkdir -p logs

_recon_run() {  # $1=glp $2=ood $3=device
    local cfg="configs/detection/ood/recon_${1}_${2}.yaml"
    python scripts/detection/evaluate_classifier.py run \
        --config="$cfg" --gpu_id=0 --device="$3"
}

echo "############### Warm shared ID cache (one config per GLP, serial) ###############"
# The first config for each GLP extracts that GLP's ID ('id_pool') + its first OOD; all
# later configs for the same GLP reuse the ID cache, so they can run concurrently safely.
for glp in "${GLPS[@]}"; do
    echo ">>> warm $glp (${OOD[0]})"
    _recon_run "$glp" "${OOD[0]}" "cuda:0"
done

echo "############### Fan out remaining recon configs across GPUs ####################"
gpu=0
pids=()
for glp in "${GLPS[@]}"; do
    for o in "${OOD[@]:1}"; do          # skip [0] (already warmed)
        dev="cuda:$((gpu % NGPU))"
        echo ">>> $glp ood:$o on $dev"
        _recon_run "$glp" "$o" "$dev" >"logs/ood_${glp}_${o}.log" 2>&1 &
        pids+=($!)
        gpu=$((gpu + 1))
        # keep at most NGPU jobs in flight
        if [ "${#pids[@]}" -ge "$NGPU" ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
    done
done
# baseline extraction (Instruct, mean-pooled) — independent, use the next GPU
echo ">>> baseline extraction on cuda:$((gpu % NGPU))"
python scripts/detection/eval_ood_baselines.py run --gpu_id=$((gpu % NGPU)) \
    >"logs/ood_baselines_extract.log" 2>&1 &
pids+=($!)
wait "${pids[@]}"
echo "[+] all extraction/scoring jobs done"

echo "############### Aggregate (CPU) ################################################"
for glp in "${GLPS[@]}"; do
    for o in "${OOD[@]}"; do
        python scripts/detection/evaluate_classifier.py aggregate \
            --out_dir="results/ood/recon-${glp}-${o}"
    done
done
python scripts/detection/eval_ood_baselines.py aggregate --out_dir=results/ood/baselines

echo "############### Best-layer summary #############################################"
python - <<'PY'
import json

OOD = ["advbench", "harmbench", "wjb_vanilla", "wjb_adversarial", "toxicchat"]


def best(path):
    try:
        bl = json.load(open(path))["aggregate"]["best_layer"]
        m = bl.get("recon_error", bl)
        return m["auprc"], m["auroc"], m.get("best_layer")
    except Exception:
        return None


print("\n=== GLP recon (best-layer AUPRC / AUROC / layer) ===")
print(f"{'ood':<18}{'new-GLP':<22}{'orig-GLP':<22}")
for o in OOD:
    n = best(f"results/ood/recon-newglp-{o}/results.json")
    g = best(f"results/ood/recon-origglp-{o}/results.json")
    fn = f"{n[0]:.3f}/{n[1]:.3f} L{n[2]}" if n else "-"
    fg = f"{g[0]:.3f}/{g[1]:.3f} L{g[2]}" if g else "-"
    print(f"{o:<18}{fn:<22}{fg:<22}")

print("\n=== Supervised baselines (best-layer AUROC; regime x eval-set) ===")
try:
    r = json.load(open("results/ood/baselines/results.json"))["regimes"]
    for method, regimes in r.items():
        print(f"\n[{method}]")
        print("  " + f"{'regime':<16}" + "".join(f"{o[:12]:<14}" for o in OOD))
        for regime, row in regimes.items():
            cells = "".join(f"{row[o]['auroc']:<14.3f}" for o in OOD)
            print(f"  {regime:<16}{cells}")
except Exception as e:
    print(f"  (no baseline results: {e})")
PY
