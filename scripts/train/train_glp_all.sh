#!/usr/bin/env bash
# Train the six single-layer GLPs for guard-glp-benign: {useronly, full} x {08, 12, 14}.
#
#   bash scripts/train/train_glp_all.sh [device]
#
# Each run reads data/llama1b-guardglpbenign-<view>/last/layer_<NN>/ (built first by
# scripts/dataset/build_activations.py) and writes runs/glp-llama1b-guardglpbenign-*.
# Override device (default cuda:0) as the first arg.
set -euo pipefail

CONFIG="configs/train/glp_llama1b_guardglpbenign.yaml"
DEVICE="${1:-cuda:0}"

for VIEW in useronly full; do
  for LAYER in 08 12 14; do
    echo "=== training GLP: view=${VIEW} layer=${LAYER} (device=${DEVICE}) ==="
    python scripts/train/train_glp.py \
      config="${CONFIG}" \
      device="${DEVICE}" \
      view="${VIEW}" \
      layer="${LAYER}"
  done
done
