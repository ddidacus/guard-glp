#!/bin/bash
# Re-run the 5 new-GLP recon configs with mean pooling (the new GLP is streamed
# all-token; last-token gave inverted recon error). Wipes stale last-token results,
# fans the 5 configs across GPUs, aggregates, then prints the analysis.
#
#   bash scripts/detection/rerun_newglp_mean.sh
set -euo pipefail
set -a; [ -f .env ] && . ./.env; set +a

OOD=(advbench harmbench wjb_vanilla wjb_adversarial toxicchat)
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
mkdir -p logs

echo "[+] wiping stale last-token new-GLP results"
for o in "${OOD[@]}"; do rm -rf "results/ood/recon-newglp-${o}"; done

echo "[+] running 5 new-GLP configs (mean-pooled) across $NGPU GPU(s)"
gpu=0
pids=()
for o in "${OOD[@]}"; do
    dev="cuda:$((gpu % NGPU))"
    echo "    newglp ood:$o on $dev"
    python scripts/detection/evaluate_classifier.py run \
        --config="configs/detection/ood/recon_newglp_${o}.yaml" --gpu_id=0 --device="$dev" \
        >"logs/rerun_newglp_${o}.log" 2>&1 &
    pids+=($!)
    gpu=$((gpu + 1))
done
wait "${pids[@]}"

echo "[+] aggregating"
for o in "${OOD[@]}"; do
    python scripts/detection/evaluate_classifier.py aggregate \
        --out_dir="results/ood/recon-newglp-${o}"
done

echo "[+] analysis"
python scripts/detection/inspect_ood_results.py
python scripts/detection/plot_ood_layers.py
echo "[+] done — figures in results/ood/figures/"
