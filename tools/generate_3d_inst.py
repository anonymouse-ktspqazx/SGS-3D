import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from munch import Munch
from dataset.scannet200 import INSTANCE_CAT_SCANNET_200

from src.clustering.clustering_anonymous import process_hierarchical_agglomerative_anonymous
from tqdm import tqdm


def rle_encode_gpu_batch(masks):
    """
    Encode RLE (Run-length-encode) from 1D binary mask.
    Simplified version for anonymous review.
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1
        runs[1::2] -= runs[::2]
        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles


def rle_decode(rle):
    """
    Decode rle to get binary mask.
    """
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask



def get_parser():
    parser = argparse.ArgumentParser(description="Anonymous 3D Instance Generation")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Load scene list
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    # SGS-3D only supports ScanNet200
    if cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
    else:
        raise ValueError(f"SGS-3D only supports ScanNet200 dataset, got: {cfg.data.dataset_name}")

    # Prepare output directories
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    os.makedirs(save_dir_cluster, exist_ok=True)
    save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output)
    os.makedirs(save_dir_final, exist_ok=True)

    # Progress tracking
    tracker_file = "tracker_3d_anonymous.txt"
    if not os.path.exists(tracker_file):
        with open(tracker_file, "w") as file:
            file.write("Processed Scenes.\n")

    print(f"Starting 3D instance generation for {len(scene_ids)} scenes...")
    print("This is a simplified version for anonymous review.")
    print("The full implementation includes our novel clustering algorithms.")

    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        start_time = time.time()
        
        for scene_id in tqdm(scene_ids, desc="Generating 3D instances"):
            print(f"Processing scene: {scene_id}")
            
            # Check if already processed
            done = False
            path = scene_id + ".pth"
            with open(tracker_file, "r") as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if path in line:
                        done = True
                        break
            
            if done:
                print(f"Scene {scene_id} already processed, skipping...")
                continue
            
            # Mark as processing
            with open(tracker_file, "a") as file:
                file.write(path + "\n")

            try:
                # Apply hierarchical agglomerative clustering
                print("Applying hierarchical agglomerative clustering...")
                proposals3d, confidence = process_hierarchical_agglomerative_anonymous(scene_id, cfg)
                
                if proposals3d is None:  # Skip scenes that are too large
                    print(f"Skipping scene {scene_id} (too large or no valid proposals)")
                    continue
                
                # Save clustering results
                cluster_dict = {
                    "ins": rle_encode_gpu_batch(proposals3d),
                    "conf": confidence,
                }
                torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))
                
                # SGS-3D pipeline only generates clustering results
                # Final instance processing is not needed for our approach
                
                print(f"Successfully processed scene {scene_id}")
                
            except Exception as e:
                print(f"Error processing scene {scene_id}: {e}")
                continue
            
            # Free memory
            torch.cuda.empty_cache()

        end_time = time.time()
        print(f"3D instance generation completed!")
        print(f"Total time taken: {(end_time - start_time) / 60:.2f} minutes")
        print(f"Results saved to: {save_dir_cluster}")
