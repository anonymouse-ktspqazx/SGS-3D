import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


def compute_projected_pts_anonymous(points, pose, intrinsic, image_dim):
    """
    Project 3D points to 2D image coordinates
    Simplified version for anonymous review
    """
    device = points.device
    n_points = points.shape[0]
    
    # Transform to camera coordinates
    points_homo = torch.cat([points, torch.ones(n_points, 1, device=device)], dim=1)
    world_to_camera = torch.inverse(pose)
    points_cam = (world_to_camera @ points_homo.T).T[:, :3]
    
    # Project to image plane
    z = points_cam[:, 2].clamp(min=1e-6)
    x_norm = points_cam[:, 0] / z
    y_norm = points_cam[:, 1] / z
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = x_norm * fx + cx
    v = y_norm * fy + cy
    
    # Check bounds
    valid = (u >= 0) & (u < image_dim[0]) & (v >= 0) & (v < image_dim[1]) & (z > 0)
    
    return torch.stack([u, v], dim=1), valid


def compute_visibility_mask_anonymous(points_2d, depths, depth_image, threshold=0.1):
    """
    Compute visibility mask for projected points
    Simplified version for anonymous review
    """
    device = points_2d.device
    n_points = len(points_2d)
    
    if depth_image is None:
        return torch.ones(n_points, dtype=torch.bool, device=device)
    
    u, v = points_2d[:, 0].long(), points_2d[:, 1].long()
    h, w = depth_image.shape
    
    # Clamp coordinates
    u = u.clamp(0, w - 1)
    v = v.clamp(0, h - 1)
    
    # Get depth values
    if isinstance(depth_image, np.ndarray):
        depth_image = torch.from_numpy(depth_image).to(device)
    
    image_depths = depth_image[v, u]
    depth_diff = torch.abs(depths - image_depths)
    
    return depth_diff < threshold


def find_connected_components_anonymous(adjacency_matrix, threshold=0.5):
    """
    Find connected components in adjacency matrix
    Simplified version for anonymous review
    """
    n_nodes = adjacency_matrix.shape[0]
    visited = torch.zeros(n_nodes, dtype=torch.bool)
    components = []
    
    for i in range(n_nodes):
        if visited[i]:
            continue
        
        # BFS to find connected component
        component = []
        queue = [i]
        visited[i] = True
        
        while queue:
            node = queue.pop(0)
            component.append(node)
            
            # Find neighbors
            neighbors = torch.where(adjacency_matrix[node] > threshold)[0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor.item())
        
        if len(component) > 1:  # Only keep components with multiple nodes
            components.append(component)
    
    return components


def resolve_overlapping_masks_anonymous(masks, iou_threshold=0.7):
    """
    Resolve overlapping masks using IoU-based filtering
    Simplified version for anonymous review
    """
    if len(masks) <= 1:
        return masks
    
    n_masks = len(masks)
    keep = torch.ones(n_masks, dtype=torch.bool)
    
    # Compute pairwise IoUs
    for i in range(n_masks):
        if not keep[i]:
            continue
            
        for j in range(i + 1, n_masks):
            if not keep[j]:
                continue
            
            # Compute IoU
            intersection = (masks[i] & masks[j]).sum().float()
            union = (masks[i] | masks[j]).sum().float()
            iou = intersection / (union + 1e-6)
            
            if iou > iou_threshold:
                # Keep the larger mask
                if masks[i].sum() >= masks[j].sum():
                    keep[j] = False
                else:
                    keep[i] = False
                    break
    
    return [masks[i] for i in range(n_masks) if keep[i]]


def compute_mask_features_anonymous(masks, features):
    """
    Compute features for each mask region
    Simplified version for anonymous review
    """
    mask_features = []
    
    for mask in masks:
        if mask.sum() == 0:
            # Empty mask
            mask_features.append(torch.zeros_like(features[0]))
            continue
        
        # Average features in mask region
        mask_feat = features[mask].mean(dim=0)
        mask_features.append(mask_feat)
    
    return torch.stack(mask_features) if mask_features else torch.empty(0, features.shape[1])


def cluster_masks_by_similarity_anonymous(masks, features, similarity_threshold=0.8):
    """
    Cluster masks based on feature similarity
    Simplified version for anonymous review
    """
    if len(masks) <= 1:
        return list(range(len(masks)))
    
    # Compute mask features
    mask_features = compute_mask_features_anonymous(masks, features)
    
    if len(mask_features) == 0:
        return []
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(mask_features, mask_features.T)
    
    # Apply clustering (simplified)
    n_masks = len(masks)
    clusters = []
    assigned = torch.zeros(n_masks, dtype=torch.bool)
    
    for i in range(n_masks):
        if assigned[i]:
            continue
        
        # Start new cluster
        cluster = [i]
        assigned[i] = True
        
        # Find similar masks
        similarities = similarity_matrix[i]
        similar_indices = torch.where(similarities > similarity_threshold)[0]
        
        for j in similar_indices:
            if not assigned[j] and j != i:
                cluster.append(j.item())
                assigned[j] = True
        
        clusters.append(cluster)
    
    return clusters


def merge_masks_in_clusters_anonymous(masks, clusters):
    """
    Merge masks within each cluster
    Simplified version for anonymous review
    """
    merged_masks = []
    
    for cluster in clusters:
        if len(cluster) == 1:
            merged_masks.append(masks[cluster[0]])
        else:
            # Merge masks in cluster
            merged_mask = masks[cluster[0]].clone()
            for idx in cluster[1:]:
                merged_mask = merged_mask | masks[idx]
            merged_masks.append(merged_mask)
    
    return merged_masks


def filter_small_masks_anonymous(masks, min_points=100):
    """
    Filter out masks with too few points
    """
    filtered_masks = []
    
    for mask in masks:
        if mask.sum() >= min_points:
            filtered_masks.append(mask)
    
    return filtered_masks


def compute_mask_overlap_matrix_anonymous(masks):
    """
    Compute overlap matrix between masks
    """
    n_masks = len(masks)
    overlap_matrix = torch.zeros(n_masks, n_masks)
    
    for i in range(n_masks):
        for j in range(n_masks):
            if i == j:
                overlap_matrix[i, j] = 1.0
            else:
                intersection = (masks[i] & masks[j]).sum().float()
                union = (masks[i] | masks[j]).sum().float()
                overlap_matrix[i, j] = intersection / (union + 1e-6)
    
    return overlap_matrix
