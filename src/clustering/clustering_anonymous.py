import itertools
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from dataset.scannet_loader import ScanNetReader, scaling_mapping
from dataset import build_dataset
from src.clustering.clustering_utils_anonymous import (
    compute_projected_pts_anonymous,
    compute_visibility_mask_anonymous,
    find_connected_components_anonymous,
    resolve_overlapping_masks_anonymous,
)
from src.fusion_util import NMS_cuda
from src.mapper import PointCloudToImageMapper
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import pickle

# Import simplified clustering algorithms
from src.clustering.dcfeature_grow_anonymous import (
    voxelize_anonymous, 
    vccs_grow_spp_anonymous, 
    vccs_grow_anonymous
)


def show_mask(mask, ax, random_color=False):
    """
    Mask visualization
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def hierarchical_agglomerative_clustering_anonymous(pcd_list, left, right, n_points, 
                                                  ious_level, level, inter, point_acc, 
                                                  iterative=True, points=None, spp=None, n_spp=None):
    """
    Anonymous implementation of hierarchical agglomerative clustering
    
    This is a simplified version of our novel clustering approach.
    The full implementation includes advanced algorithms that will be
    released upon paper acceptance.
    
    Args:
        pcd_list: List of point cloud data with masks and mappings
        left, right: Range indices for processing
        n_points: Number of points in the scene
        ious_level: IoU threshold for clustering
        level: Current level in hierarchy
        inter: Intersection threshold
        point_acc: Point accumulation threshold
        iterative: Whether to use iterative clustering
        points: 3D point coordinates
        spp: Superpoint assignments
        n_spp: Number of superpoints
        
    Returns:
        cluster_permask: List of clustered masks
    """
    
    if left > right:
        return []
    
    if left == right:
        device = 'cuda'
        index = left

        if pcd_list[index]["masks"] is None:
            return []
        
        masks = pcd_list[index]["masks"].cuda()
        mapping = pcd_list[index]["mapping"].cuda()
        image_dim_hw = pcd_list[index]["image_dim"]

        # Simplified mask processing
        mask3d = []
        highlight_indices = set()

        for m, mask in enumerate(masks):
            # Resize mask if needed
            if mask.shape != image_dim_hw:
                resized_mask = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0).float(),
                    size=image_dim_hw,
                    mode='nearest' 
                )
                mask = resized_mask.squeeze(0).squeeze(0).bool()

            # Get valid point indices
            idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
            if len(idx) == 0:
                continue

            # Project to 2D and check visibility
            u, v = mapping[idx, 1], mapping[idx, 0]
            valid_uv = (u >= 0) & (u < image_dim_hw[1]) & (v >= 0) & (v < image_dim_hw[0])
            
            if valid_uv.sum() == 0:
                continue

            idx_valid = idx[valid_uv]
            u_valid, v_valid = u[valid_uv], v[valid_uv]

            # Check mask coverage
            mask_values = mask[v_valid, u_valid]
            highlighted_idx = idx_valid[mask_values]
            
            if len(highlighted_idx) < 50:  # Minimum points threshold
                continue

            highlight_indices.update(highlighted_idx.cpu().numpy())

            # Create 3D mask
            group_tmp = torch.zeros(n_points, dtype=torch.int8, device=device)
            group_tmp[highlighted_idx] = 1
            mask3d.append(group_tmp)

        if not mask3d:
            return []

        # Apply simplified clustering
        print(f"Applying simplified clustering to {len(mask3d)} masks...")
        
        # Convert highlight indices to tensor
        if highlight_indices:
            highlight_points = torch.tensor(list(highlight_indices), device=device)
            
            # Apply our simplified clustering algorithm
            cluster_permask = vccs_grow_spp_anonymous(
                highlight_points, points, spp, n_spp, None, None
            )
            
            return cluster_permask
        else:
            return []

    else:
        # Recursive clustering for multiple frames
        mid = (left + right) // 2
        
        # Process left and right halves
        left_clusters = hierarchical_agglomerative_clustering_anonymous(
            pcd_list, left, mid, n_points, ious_level, level, inter, point_acc,
            iterative, points, spp, n_spp
        )
        
        right_clusters = hierarchical_agglomerative_clustering_anonymous(
            pcd_list, mid + 1, right, n_points, ious_level, level, inter, point_acc,
            iterative, points, spp, n_spp
        )
        
        # Merge clusters (simplified version)
        all_clusters = left_clusters + right_clusters
        
        if not all_clusters:
            return []
        
        # Apply simple merging based on overlap
        merged_clusters = merge_overlapping_clusters_anonymous(all_clusters, ious_level)
        
        return merged_clusters


def merge_overlapping_clusters_anonymous(clusters, iou_threshold=0.5):
    """
    Merge overlapping clusters based on IoU
    Simplified version for anonymous review
    """
    if len(clusters) <= 1:
        return clusters
    
    # Compute pairwise IoUs
    n_clusters = len(clusters)
    iou_matrix = torch.zeros(n_clusters, n_clusters)
    
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            intersection = (clusters[i] & clusters[j]).sum().float()
            union = (clusters[i] | clusters[j]).sum().float()
            iou = intersection / (union + 1e-6)
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    
    # Find clusters to merge
    merged = [False] * n_clusters
    final_clusters = []
    
    for i in range(n_clusters):
        if merged[i]:
            continue
            
        # Start a new merged cluster
        current_cluster = clusters[i].clone()
        merged[i] = True
        
        # Find all clusters that should be merged with this one
        for j in range(i + 1, n_clusters):
            if not merged[j] and iou_matrix[i, j] > iou_threshold:
                current_cluster = current_cluster | clusters[j]
                merged[j] = True
        
        final_clusters.append(current_cluster)
    
    return final_clusters


def process_hierarchical_agglomerative_anonymous(scene_id, cfg):
    """
    Process a single scene using hierarchical agglomerative clustering
    
    This is the main entry point for our anonymous clustering method.
    The full implementation includes significant algorithmic improvements.
    
    Args:
        scene_id: Scene identifier
        cfg: Configuration object
        
    Returns:
        proposals3d: 3D instance proposals
        confidence: Confidence scores for proposals
    """
    
    print(f"Processing scene {scene_id} with anonymous clustering method...")
    
    # Load scene data
    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    loader = build_dataset(root_path=scene_dir, cfg=cfg)
    
    # Load point cloud
    points = loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]
    
    print(f"Loaded point cloud with {n_points} points")
    
    # Check scene size limits
    if n_points > 1000000:  # Skip very large scenes
        print(f"Scene {scene_id} too large ({n_points} points), skipping...")
        return None, None
    
    # Load superpoints
    spp_path = os.path.join(cfg.data.spp_path, f"{scene_id}.pth")
    if os.path.exists(spp_path):
        spp_data = torch.load(spp_path)
        spp = torch.from_numpy(spp_data).cuda()
        n_spp = spp.max().item() + 1
    else:
        print(f"Warning: No superpoints found for {scene_id}, using point-level clustering")
        spp = torch.arange(n_points, device='cuda')
        n_spp = n_points
    
    # Load 2D masks and mappings
    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)
    mask2d_path = os.path.join(exp_path, cfg.exp.mask2d_output, f"{scene_id}.pth")
    
    if not os.path.exists(mask2d_path):
        print(f"Error: 2D masks not found for {scene_id}")
        return None, None
    
    mask2d_data = torch.load(mask2d_path)
    
    # Initialize point cloud mapper
    img_dim = cfg.data.img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, 
        intrinsics=loader.global_intrinsic, 
        cut_bound=cfg.data.cut_num_pixel_boundary
    )
    
    # Process frames and create pcd_list
    pcd_list = []
    frame_count = 0
    
    print("Processing frames and computing mappings...")
    
    for i in trange(0, len(loader), cfg.data.img_interval):
        frame = loader[i]
        frame_id = frame["frame_id"]
        
        if frame_id not in mask2d_data:
            continue
        
        # Load frame data
        pose = loader.read_pose(frame["pose_path"])
        depth = loader.read_depth(frame["depth_path"])
        
        # Compute point-to-image mapping
        mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
        mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(
            pose, points, depth, intrinsic=frame.get("translated_intrinsics")
        )
        
        # Load masks for this frame
        frame_masks = mask2d_data[frame_id]
        if "masks" not in frame_masks or len(frame_masks["masks"]) == 0:
            continue
        
        # Convert RLE masks to tensors
        masks_rle = frame_masks["masks"]
        masks = []
        for mask_rle in masks_rle:
            # Simplified RLE decoding
            mask = torch.zeros(img_dim[1], img_dim[0], dtype=torch.bool)
            # In full implementation, proper RLE decoding would be here
            masks.append(mask)
        
        if masks:
            masks_tensor = torch.stack(masks)
            
            pcd_list.append({
                "masks": masks_tensor,
                "mapping": mapping,
                "image_dim": (img_dim[1], img_dim[0]),
                "frame_id": frame_id
            })
            
            frame_count += 1
    
    if not pcd_list:
        print(f"No valid frames found for scene {scene_id}")
        return None, None
    
    print(f"Processing {frame_count} frames with hierarchical clustering...")
    
    # Apply hierarchical agglomerative clustering
    try:
        cluster_permask = hierarchical_agglomerative_clustering_anonymous(
            pcd_list=pcd_list,
            left=0,
            right=len(pcd_list) - 1,
            n_points=n_points,
            ious_level=cfg.cluster.visi,
            level=0,
            inter=cfg.cluster.recall,
            point_acc=cfg.cluster.point_visi,
            iterative=True,
            points=points,
            spp=spp,
            n_spp=n_spp
        )
        
        if not cluster_permask:
            print(f"No clusters generated for scene {scene_id}")
            return None, None
        
        # Convert to final format
        proposals3d = torch.stack(cluster_permask, dim=0)
        confidence = torch.ones(len(cluster_permask)) * 0.8  # Simplified confidence
        
        print(f"Generated {len(cluster_permask)} 3D instance proposals")
        
        return proposals3d, confidence
        
    except Exception as e:
        print(f"Error in clustering for scene {scene_id}: {e}")
        return None, None
