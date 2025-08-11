import os
import yaml
import torch
import argparse
import numpy as np
from munch import Munch
from tqdm import tqdm, trange
import cv2
import pickle

# Util
from util2d.grounding_dino_sam_anonymous import GroundingDINO_SAM_Anonymous
from util2d.util import masks_to_rle

from dataset.scannet200 import INSTANCE_CAT_SCANNET_200

############################################## 2D Foundation Models + SAM ##############################################
"""
Anonymous implementation of 2D mask extraction using foundation models.
This is a simplified version of our approach for anonymous review.
"""

def get_parser():
    parser = argparse.ArgumentParser(description="Anonymous 3D Instance Segmentation - 2D Mask Extraction")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Load scene split
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    # SGS-3D only supports ScanNet200
    if cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
    else:
        raise ValueError(f"SGS-3D only supports ScanNet200 dataset, got: {cfg.data.dataset_name}")

    # Initialize foundation model
    print("Loading foundation models...")
    model = GroundingDINO_SAM_Anonymous(cfg)

    # Create output directories
    save_dir = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)
    save_dir_feat = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.grounded_feat_output)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_feat, exist_ok=True)

    # Process each scene
    print(f"Processing {len(scene_ids)} scenes...")
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids, desc="Extracting 2D masks"):
            # Check if already processed
            output_path = os.path.join(save_dir, scene_id + ".pth")
            if os.path.exists(output_path):
                print(f"Scene {scene_id} already processed, skipping...")
                continue

            print(f"Processing scene: {scene_id}")
            
            # Generate 2D masks and features
            grounded_data_dict, grounded_features = model.extract_masks_and_features(
                scene_id,
                class_names,
                cfg=cfg,
            )

            # Save results
            torch.save({"feat": grounded_features}, os.path.join(save_dir_feat, scene_id + ".pth"))
            torch.save(grounded_data_dict, os.path.join(save_dir, scene_id + ".pth"))
            
            # Free memory
            torch.cuda.empty_cache()
            
    print("2D mask extraction completed!")
