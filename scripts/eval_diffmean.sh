#!/bin/bash
#SBATCH -J ift6164-diffmean
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

CONFIG="${1:-configs/eval_diffmean.yaml}"

OUT_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['out_dir'])")

# Pass 1: extract activations in parallel across GPUs
PID_LIST=""
for gpu_id in 0 1 2 3; do
    echo "Launching activation extraction on GPU $gpu_id"
    python eval_diffmean.py run --config="$CONFIG" --gpu_id="$gpu_id" &
    PID_LIST+=" $!"
    sleep 5
done
trap "kill $PID_LIST" SIGINT
echo "Extracting activations..."
wait $PID_LIST
echo "All GPU jobs finished."

# Pass 2: compute steering vector + evaluate (no GPU needed)
echo "Computing DiffMean steering vector and evaluating..."
python eval_diffmean.py aggregate --out_dir="$OUT_DIR"

echo "Done."
