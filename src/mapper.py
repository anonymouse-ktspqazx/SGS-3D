import torch
import numpy as np
import cv2

class PointCloudToImageMapper:
    """
    Simplified point cloud to image mapper for anonymous review
    """
    
    def __init__(self, image_dim, intrinsics, cut_bound=0):
        self.image_dim = image_dim  # [width, height]
        self.intrinsics = intrinsics
        self.cut_bound = cut_bound
        
    def compute_mapping_torch(self, pose, points, depth, intrinsic=None):
        """
        Compute mapping from 3D points to 2D image coordinates
        
        Args:
            pose: Camera pose [4, 4]
            points: 3D points [N, 3]
            depth: Depth image
            intrinsic: Camera intrinsic matrix [3, 3]
            
        Returns:
            mapping: Point to pixel mapping [N, 3] (x, y, valid)
        """
        if intrinsic is None:
            intrinsic = self.intrinsics
            
        device = points.device
        n_points = points.shape[0]
        
        # Transform points to camera coordinate system
        points_homo = torch.cat([points, torch.ones(n_points, 1, device=device)], dim=1)
        world_to_camera = torch.inverse(pose)
        points_cam = (world_to_camera @ points_homo.T).T[:, :3]
        
        # Project to image plane
        points_2d = self.project_points(points_cam, intrinsic)
        
        # Check visibility
        valid_mask = self.check_visibility(points_2d, points_cam[:, 2], depth)
        
        # Create mapping
        mapping = torch.zeros(n_points, 3, dtype=torch.long, device=device)
        mapping[:, :2] = points_2d.long()
        mapping[:, 2] = valid_mask.long()
        
        return mapping
    
    def project_points(self, points_cam, intrinsic):
        """
        Project 3D camera coordinates to 2D image coordinates
        
        Args:
            points_cam: Points in camera coordinates [N, 3]
            intrinsic: Camera intrinsic matrix [3, 3]
            
        Returns:
            points_2d: 2D image coordinates [N, 2]
        """
        # Avoid division by zero
        z = points_cam[:, 2].clamp(min=1e-6)
        
        # Project to normalized image coordinates
        x_norm = points_cam[:, 0] / z
        y_norm = points_cam[:, 1] / z
        
        # Apply intrinsic parameters
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        u = x_norm * fx + cx
        v = y_norm * fy + cy
        
        points_2d = torch.stack([u, v], dim=1)
        return points_2d
    
    def check_visibility(self, points_2d, depths, depth_image, depth_threshold=0.1):
        """
        Check if points are visible in the image
        
        Args:
            points_2d: 2D image coordinates [N, 2]
            depths: Point depths [N]
            depth_image: Depth image
            depth_threshold: Depth difference threshold
            
        Returns:
            valid_mask: Boolean mask for visible points [N]
        """
        device = points_2d.device
        n_points = len(points_2d)
        
        # Check image bounds
        u, v = points_2d[:, 0], points_2d[:, 1]
        h, w = self.image_dim[1], self.image_dim[0]
        
        bounds_mask = (u >= self.cut_bound) & (u < w - self.cut_bound) & \
                     (v >= self.cut_bound) & (v < h - self.cut_bound) & \
                     (depths > 0)
        
        if depth_image is None:
            return bounds_mask
        
        # Check depth consistency
        depth_mask = torch.zeros(n_points, dtype=torch.bool, device=device)
        
        valid_indices = torch.where(bounds_mask)[0]
        if len(valid_indices) > 0:
            u_valid = u[valid_indices].long().clamp(0, w-1)
            v_valid = v[valid_indices].long().clamp(0, h-1)
            
            # Get depth values from depth image
            if isinstance(depth_image, np.ndarray):
                depth_image = torch.from_numpy(depth_image).to(device)
            
            image_depths = depth_image[v_valid, u_valid]
            point_depths = depths[valid_indices]
            
            # Check if point depth is close to image depth
            depth_diff = torch.abs(point_depths - image_depths)
            depth_valid = depth_diff < depth_threshold
            
            depth_mask[valid_indices] = depth_valid
        
        return bounds_mask & depth_mask
