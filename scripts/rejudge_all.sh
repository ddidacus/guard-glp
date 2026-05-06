#!/bin/bash

for dir in \
    results/results_steering/steering_none \
    results/results_steering/steering_sv \
    results/results_steering/steering_glp \
    results/results_steering/steering_refusal_none \
    results/results_steering/steering_refusal_sv \
    results/results_steering/steering_refusal_glp \
; do
    echo "========== Rejudging $dir =========="
    RESULTS_DIR="$dir" bash scripts/rejudge.sh
done