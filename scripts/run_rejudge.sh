#!/bin/bash
export HF_HOME=$SCRATCH/.cache
export UV_CACHE_DIR=$SCRATCH/.cache

source .venv/bin/activate

python scripts/rejudge_responses.py "$@"
