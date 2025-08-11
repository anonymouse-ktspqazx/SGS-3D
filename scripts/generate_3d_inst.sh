#!/bin/bash

# Anonymous 3D Instance Segmentation - 3D Instance Generation
# This script generates 3D instances from 2D masks using our novel clustering approach

dataset_cfg=${1:-'configs/scannet200_sgs3d.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

echo "Starting 3D instance generation with config: $dataset_cfg"
echo "This step lifts 2D masks to 3D instances using our novel clustering method"

CUDA_VISIBLE_DEVICES=0 python3 tools/generate_3d_inst.py --config $dataset_cfg
