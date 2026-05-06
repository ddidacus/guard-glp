#!/bin/bash

echo "================ Evaluating: linear probe ================ "
bash scripts/evaluate_linear_probe.sh configs/paper/eval_lp.yaml

echo "================ Evaluating: GLP-PI ================ "
bash scripts/evaluate_classifier.sh configs/paper/eval_pi.yaml

echo "================ Evaluating: DTE ================ "
bash scripts/evaluate_classifier.sh configs/paper/eval_dte.yaml

echo "================ Evaluating: GLP-DTE ================ "
bash scripts/evaluate_classifier.sh configs/paper/eval_dte_glp.yaml

echo "================ Evaluating: DiffMean ================ "
bash scripts/evaluate_diffmean.sh configs/paper/eval_diffmean.yaml