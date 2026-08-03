#!/bin/bash
#SBATCH -J glp-dataset-prep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:40gb:4
#SBATCH --partition=short-unkillable
#SBATCH --time=3:00:00
#SBATCH --output=logs/merge_train_sets_%j.out
#SBATCH --error=logs/merge_train_sets_%j.err
set -euo pipefail

source .cluster_env
source .venv/bin/activate

# Number of shard workers == number of GPUs to parallelize over; each worker
# is pinned to one GPU index via CUDA_VISIBLE_DEVICES and independently
# filters/formats/decontaminates its own slice of the corpus.
NUM_THREADS="${NUM_THREADS:-4}"
SHARD_DIR="${SHARD_DIR:-data/guardglp_benign_shards}"
mkdir -p "$SHARD_DIR" logs

# Embed the WildJailbreak reference ONCE (on GPU 0) and cache it, rather than
# re-embedding it in each of the NUM_THREADS shard workers (~8x GPU savings).
echo "[+] Embedding reference once → $SHARD_DIR/wildjb_ref_emb.pt"
CUDA_VISIBLE_DEVICES=0 python scripts/preprocessing/merge_train_sets.py embed_reference \
    --shard_dir "$SHARD_DIR" \
    --embed_batch_size 256 \
    2>&1 | tee "logs/merge_train_sets_embed_reference.log"

echo "[+] Sharding across $NUM_THREADS GPU(s) → $SHARD_DIR"
pids=()
for ((i = 0; i < NUM_THREADS; i++)); do
    CUDA_VISIBLE_DEVICES="$i" python scripts/preprocessing/merge_train_sets.py shard \
        --shard_id "$i" \
        --num_shards "$NUM_THREADS" \
        --shard_dir "$SHARD_DIR" \
        --embed_batch_size 256 \
        --sim_threshold 0.99 \
        > "logs/merge_train_sets_shard_${i}.log" 2>&1 &
    pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
    echo "[!] One or more shard workers failed; see logs/merge_train_sets_shard_*.log" >&2
    exit 1
fi

echo "[+] All shards done — merging and pushing …"
python scripts/preprocessing/merge_train_sets.py finalize \
    --shard_dir "$SHARD_DIR" \
    --num_shards "$NUM_THREADS" \
    --output_dir data/guardglp_benign \
    --sim_threshold 0.99 \
    --semantic_dedup_samples_per_bucket 512 \
    --embed_batch_size 256 \
    --push_to_hub --private
