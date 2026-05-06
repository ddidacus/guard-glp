#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

CONFIG="${1:-configs/paper/eval_pi.yaml}"
NUM_GPUS=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG')).get('num_gpus', 4))")

PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching eval_classifier on GPU $gpu_id"
    python scripts/evaluate_classifier.py run --config="$CONFIG" --gpu_id="$gpu_id" &
    PID_LIST+=" $!"
    sleep 5
done
trap "kill $PID_LIST" SIGINT
echo "Computing log-probs..."
wait $PID_LIST

echo "Aggregating results..."
python scripts/evaluate_classifier.py aggregate --out_dir="$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['out_dir'])")"

echo "Done."
