#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:80gb:4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00

MODEL_ID="google/gemma-4-31B"
KEY="g3mm4"
N_GPUS=4

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache
source .venv/bin/activate

vllm serve $MODEL_ID \
  --dtype auto \
  --api-key $KEY \
  --tensor-parallel-size $N_GPUS \
  --max-model-len 16536 \
  --gpu-memory-utilization 0.85 \
  --max-num-batched-tokens 16536