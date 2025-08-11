#!/bin/bash

# Anonymous 3D Instance Segmentation - 2D Mask Extraction
# This script extracts 2D masks from RGB-D sequences

dataset_cfg=${1:-'configs/scannet200_sgs3d.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

echo "Starting 2D mask extraction with config: $dataset_cfg"
CUDA_VISIBLE_DEVICES=0 python3 tools/extract_2d_masks.py --config $dataset_cfg
