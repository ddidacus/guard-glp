#!/bin/bash
#SBATCH -J ift6164-diffmean
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=32G
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

CONFIG="${1:-configs/eval_diffmean.yaml}"

OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['out_dir'])")
NUM_GPUS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('num_gpus', 4))")

# Pass 1: extract activations in parallel across GPUs
PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching activation extraction on GPU $gpu_id"
    python scripts/evaluate_diffmean.py run --config="$CONFIG" --gpu_id="$gpu_id" &
    PID_LIST+=" $!"
    sleep 1
done
trap "kill $PID_LIST" SIGINT
echo "Extracting activations..."
wait $PID_LIST
echo "All GPU jobs finished."

# Pass 2: compute steering vector + evaluate (no GPU needed)
echo "Computing DiffMean steering vector and evaluating..."
python scripts/evaluate_diffmean.py aggregate --out_dir="$OUT_DIR"

echo "Done."
