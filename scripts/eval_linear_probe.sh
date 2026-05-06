#!/bin/bash
#SBATCH -J ift6164-linear-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:40gb:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=32G
#SBATCH --partition=long
#SBATCH --time=2:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

CONFIG="${1:-configs/eval_lp.yaml}"

# Read fields from config via Python
OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['out_dir'])")
PROBE_LR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('probe_lr', 1e-3))")
PROBE_EPOCHS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('probe_epochs', 100))")
PROBE_WD=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('probe_wd', 1e-4))")
PROBE_BATCH=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('probe_batch_size', 64))")
PROBE_DEVICE=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('probe_device', 'cpu'))")

# Pass 1: extract activations in parallel across GPUs
PID_LIST=""
for gpu_id in 0 1 2 3 4 5 6 7; do
    echo "Launching activation extraction on GPU $gpu_id"
    python eval_linear_probe.py run --config="$CONFIG" --gpu_id="$gpu_id" &
    PID_LIST+=" $!"
    sleep 1
done
trap "kill $PID_LIST" SIGINT
echo "Extracting activations..."
wait $PID_LIST
echo "All GPU jobs finished."

# Pass 2: train probes + evaluate (single job, no GPU needed)
echo "Training probes and computing metrics..."
python eval_linear_probe.py aggregate \
    --out_dir="$OUT_DIR" \
    --probe_lr="$PROBE_LR" \
    --probe_epochs="$PROBE_EPOCHS" \
    --probe_wd="$PROBE_WD" \
    --probe_batch_size="$PROBE_BATCH" \
    --device="$PROBE_DEVICE"

echo "Done."
