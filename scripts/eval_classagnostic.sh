#!/bin/bash

# Anonymous 3D Instance Segmentation - Class-Agnostic Evaluation
# This script evaluates the generated 3D instances using class-agnostic metrics

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

dataset_cfg=${1:-'configs/scannet200_sgs3d.yaml'}

echo "Starting class-agnostic evaluation with config: $dataset_cfg"
echo "This evaluation uses our enhanced class-agnostic metrics"

CUDA_VISIBLE_DEVICES=0 python3 tools/eval_classagnostic.py --config $dataset_cfg --type 2D
