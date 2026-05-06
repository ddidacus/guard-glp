#!/bin/bash
#SBATCH -J ift6164-vizact
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --constraint=ampere|lovelace|hopper
#SBATCH --mem=64G
#SBATCH --partition=long
#SBATCH --time=4:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
export PYTHONUNBUFFERED=1

source .venv/bin/activate

CONFIG="configs/paper/eval_plotting.yaml"

_py() { python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print($1)"; }
OUT_DIR=$(_py "c['out_dir']")
METHOD=$(_py "c['method']")
LAYERS_COMPACT=$(_py "','.join(str(x) for x in c['layers'])")

# Warm the HF dataset cache serially so parallel workers don't race on file locks
echo "Warming dataset cache..."
python -c "
from datasets import load_dataset
load_dataset('ddidacus/guard-glp-data', split='train')
"

# Part 1: shard extraction (4 GPUs in parallel)
PID_LIST=""
for gpu_id in 0 1 2 3; do
    echo "Launching shard extraction on GPU $gpu_id"
    python scripts/visualize_activations.py --config="$CONFIG" --gpu_id="$gpu_id" &
    PID_LIST+=" $!"
    sleep 5
done
trap "kill $PID_LIST" SIGINT
echo "Extracting tensors..."
wait $PID_LIST

# Part 2: aggregate and plot on the main thread
echo "Extraction completed, aggregating and plotting..."
python scripts/visualize_activations.py --aggregate \
    --results_dir="$OUT_DIR" \
    --layers="$LAYERS_COMPACT" \
    --method="$METHOD"
echo "Done."
