#!/bin/bash

# ========= Refusal - Compliance
STEERING_TYPE=none OUT_DIR=results/steering_refusal_none bash scripts/steering_refusal.sh

STEERING_TYPE=sv ALPHAS="-0.01,-0.05,-0.1,-1.0,-2.0,0.01,0.05,0.1,1.0,2.0" OUT_DIR=results/steering_refusal_sv bash scripts/steering_refusal.sh

STEERING_TYPE=glp ALPHAS="-0.01,-0.05,-0.1,-1.0,-2.0,0.01,0.05,0.1,1.0,2.0" OUT_DIR=results/steering_refusal_glp bash scripts/steering_refusal.sh
