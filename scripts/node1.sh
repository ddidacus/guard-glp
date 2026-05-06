#!/bin/bash

# ========= Benign - Malicious
STEERING_TYPE=glp OUT_DIR=results/steering_glp ALPHAS="-0.01,-0.05,-0.1,-1.0,-2.0,0.01,0.05,0.1,1.0,2.0" bash scripts/steering.sh

STEERING_TYPE=none OUT_DIR=results/steering_none bash scripts/steering.sh

STEERING_TYPE=sv ALPHAS="-0.01,-0.05,-0.1,-1.0,-2.0,0.01,0.05,0.1,1.0,2.0" OUT_DIR=results/steering_sv bash scripts/steering.sh