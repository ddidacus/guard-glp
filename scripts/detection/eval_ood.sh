#!/bin/bash
#SBATCH -J ift6164-ood-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=64G
#SBATCH --partition=short-unkillable
#SBATCH --time=6:00:00

set -e

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

CONFIG="${1:-configs/detection/eval_reconstruction_err.yaml}"
LP_CONFIG="${2:-configs/detection/eval_lp.yaml}"

NUM_GPUS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('num_gpus', 4))")
OOD_RE_DIR="results/ood_glp-re"
OOD_LP_DIR="results/ood_linear-probe"

# ── GLP reconstruction error on OOD ──────────────────────────────────────────
echo "=== OOD GLP-RE evaluation ==="

PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching ood_evaluate_classifier on GPU $gpu_id"
    python scripts/detection/ood_evaluate_classifier.py run --config="$CONFIG" --gpu_id="$gpu_id" --out_dir="$OOD_RE_DIR" &
    PID_LIST+=" $!"
    sleep 5
done
trap "kill $PID_LIST 2>/dev/null" SIGINT
wait $PID_LIST

echo "Aggregating GLP-RE results..."
python scripts/detection/ood_evaluate_classifier.py aggregate --out_dir="$OOD_RE_DIR"

# ── Linear probe on OOD ─────────────────────────────────────────────────────
echo ""
echo "=== OOD Linear Probe evaluation ==="

LP_NUM_GPUS=$(python -c "import yaml; print(yaml.safe_load(open('$LP_CONFIG')).get('num_gpus', 4))")
PROBE_LR=$(python -c "import yaml; print(yaml.safe_load(open('$LP_CONFIG')).get('probe_lr', 1e-3))")
PROBE_EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('$LP_CONFIG')).get('probe_epochs', 100))")
PROBE_WD=$(python -c "import yaml; print(yaml.safe_load(open('$LP_CONFIG')).get('probe_wd', 1e-4))")
PROBE_BATCH=$(python -c "import yaml; print(yaml.safe_load(open('$LP_CONFIG')).get('probe_batch_size', 64))")
PROBE_DEVICE=$(python -c "import yaml; print(yaml.safe_load(open('$LP_CONFIG')).get('probe_device', 'cpu'))")

PID_LIST=""
for gpu_id in $(seq 0 $((LP_NUM_GPUS - 1))); do
    echo "Launching ood_eval_linear_probe on GPU $gpu_id"
    python scripts/detection/ood_eval_linear_probe.py run --config="$LP_CONFIG" --gpu_id="$gpu_id" --out_dir="$OOD_LP_DIR" &
    PID_LIST+=" $!"
    sleep 5
done
trap "kill $PID_LIST 2>/dev/null" SIGINT
wait $PID_LIST

echo "Training probes and computing metrics..."
python scripts/detection/ood_eval_linear_probe.py aggregate \
    --out_dir="$OOD_LP_DIR" \
    --probe_lr="$PROBE_LR" \
    --probe_epochs="$PROBE_EPOCHS" \
    --probe_wd="$PROBE_WD" \
    --probe_batch_size="$PROBE_BATCH" \
    --device="$PROBE_DEVICE"

echo ""
echo "=== Done ==="
echo "GLP-RE results: $OOD_RE_DIR/results.json"
echo "Linear probe results: $OOD_LP_DIR/results.json"
