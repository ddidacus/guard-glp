#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

export HF_HOME=/home/p.greiner/workspace/adv-glp/.cache
export UV_CACHE_DIR=/home/p.greiner/workspace/adv-glp/.cache

source .venv/bin/activate

NUM_GPUS="${NUM_GPUS:-8}"
MODEL="${MODEL:-1b}"
STEERING_TYPE="${STEERING_TYPE:-none}"
ALPHAS="${ALPHAS:-0.1}"
OUT_DIR="${OUT_DIR:-results/steering_refusal}"

# build --alphas flag: "0.1,0.5,1.0" -> '[0.1,0.5,1.0]'
ALPHAS_JSON="[${ALPHAS}]"

PID_LIST=""
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Launching steering_refusal on GPU $gpu_id"
    python scripts/steering_refusal.py run \
        --gpu_id="$gpu_id" \
        --num_gpus="$NUM_GPUS" \
        --model="$MODEL" \
        --steering_type="$STEERING_TYPE" \
        --alphas="$ALPHAS_JSON" \
        --out_dir="$OUT_DIR" &
    PID_LIST+=" $!"
    sleep 0.5
done
trap "kill $PID_LIST" SIGINT
echo "Generating and judging..."
wait $PID_LIST

echo "Aggregating results..."
python scripts/steering_refusal.py aggregate --out_dir="$OUT_DIR"

echo "Done."
