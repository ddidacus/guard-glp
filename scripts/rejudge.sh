#!/bin/bash

export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to the path containing *responses*.json files}"

python scripts/rejudge_responses.py --results_dir="$RESULTS_DIR"
