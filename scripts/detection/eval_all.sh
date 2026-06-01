#!/bin/bash

echo "================ Evaluating: linear probe ================ "
bash scripts/detection/eval_linear_probe.sh configs/detection/eval_lp.yaml

echo "================ Evaluating: GLP-PI ================ "
bash scripts/detection/eval_classifier.sh configs/detection/eval_pi.yaml

echo "================ Evaluating: DTE ================ "
bash scripts/detection/eval_classifier.sh configs/detection/eval_dte.yaml

echo "================ Evaluating: GLP-DTE ================ "
bash scripts/detection/eval_classifier.sh configs/detection/eval_dte_glp.yaml

echo "================ Evaluating: DiffMean ================ "
bash scripts/detection/eval_diffmean.sh configs/detection/eval_diffmean.yaml
