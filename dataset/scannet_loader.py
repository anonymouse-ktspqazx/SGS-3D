import os
import numpy as np
import torch
import cv2
from PIL import Image

class ScanNetReader:
    """
    Simplified ScanNet dataset loader for anonymous review
    """
    
    def __init__(self, root_path, cfg):
        self.root_path = root_path
        self.cfg = cfg
        
        # Load frame information
        self.color_dir = os.path.join(root_path, "color")
        self.depth_dir = os.path.join(root_path, "depth")
        self.pose_dir = os.path.join(root_path, "pose")
        self.intrinsic_dir = os.path.join(root_path, "intrinsic")
        
        # Get frame list
        if os.path.exists(self.color_dir):
            self.frame_ids = sorted([f.split('.')[0] for f in os.listdir(self.color_dir) 
                                   if f.endswith(('.jpg', '.png'))])
        else:
            self.frame_ids = []
        
        # Load global intrinsic
        self.global_intrinsic = self.load_global_intrinsic()
        
        print(f"Loaded ScanNet scene with {len(self.frame_ids)} frames")
    
    def __len__(self):
        return len(self.frame_ids)
    
    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        
        # Construct file paths
        color_path = os.path.join(self.color_dir, f"{frame_id}.jpg")
        depth_path = os.path.join(self.depth_dir, f"{frame_id}.png")
        pose_path = os.path.join(self.pose_dir, f"{frame_id}.txt")
        intrinsic_path = os.path.join(self.intrinsic_dir, f"{frame_id}.txt")
        
        # Alternative extensions
        if not os.path.exists(color_path):
            color_path = os.path.join(self.color_dir, f"{frame_id}.png")
        
        frame_data = {
            "frame_id": frame_id,
            "image_path": color_path,
            "depth_path": depth_path,
            "pose_path": pose_path,
            "intrinsic_path": intrinsic_path,
            "translated_intrinsics": self.global_intrinsic,
            "scannet_depth_intrinsic": self.global_intrinsic
        }
        
        return frame_data
    
    def read_pointcloud(self):
        """Read point cloud from PLY file"""
        ply_path = os.path.join(self.root_path, f"{os.path.basename(self.root_path)}.ply")
        
        if not os.path.exists(ply_path):
            # Alternative naming
            ply_files = [f for f in os.listdir(self.root_path) if f.endswith('.ply')]
            if ply_files:
                ply_path = os.path.join(self.root_path, ply_files[0])
            else:
                raise FileNotFoundError(f"No PLY file found in {self.root_path}")
        
        # Simple PLY reader (simplified for anonymous review)
        points = self.read_ply_simple(ply_path)
        return points
    
    def read_ply_simple(self, ply_path):
        """Simplified PLY reader"""
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
        except ImportError:
            # Fallback to manual parsing
            points = self.parse_ply_manual(ply_path)
        
        return points
    
    def parse_ply_manual(self, ply_path):
        """Manual PLY parsing as fallback"""
        points = []
        with open(ply_path, 'r') as f:
            lines = f.readlines()
            
            # Find vertex count
            vertex_count = 0
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[-1])
                elif line.startswith('end_header'):
                    header_end = i + 1
                    break
            
            # Read vertices
            for i in range(header_end, header_end + vertex_count):
                coords = lines[i].strip().split()[:3]
                points.append([float(x) for x in coords])
        
        return np.array(points)
    
    def read_image(self, image_path):
        """Read RGB image"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def read_depth(self, depth_path):
        """Read depth image"""
        if depth_path.endswith('.png'):
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) / 1000.0  # Convert to meters
        else:
            # Handle other formats if needed
            depth = np.load(depth_path) if depth_path.endswith('.npy') else None
        
        return depth
    
    def read_pose(self, pose_path):
        """Read camera pose"""
        if os.path.exists(pose_path):
            pose = np.loadtxt(pose_path)
            if pose.shape == (16,):
                pose = pose.reshape(4, 4)
        else:
            # Return identity if pose not available
            pose = np.eye(4)
        
        return torch.from_numpy(pose).float()
    
    def load_global_intrinsic(self):
        """Load global camera intrinsic parameters"""
        # Try to load from intrinsic directory
        if os.path.exists(self.intrinsic_dir):
            intrinsic_files = os.listdir(self.intrinsic_dir)
            if intrinsic_files:
                intrinsic_path = os.path.join(self.intrinsic_dir, intrinsic_files[0])
                intrinsic = np.loadtxt(intrinsic_path)
                if intrinsic.shape == (9,):
                    intrinsic = intrinsic.reshape(3, 3)
                elif intrinsic.shape == (16,):
                    intrinsic = intrinsic.reshape(4, 4)[:3, :3]
                return torch.from_numpy(intrinsic).float()
        
        # Default intrinsic for ScanNet
        img_dim = self.cfg.data.img_dim
        fx = fy = 525.0  # Default focal length
        cx, cy = img_dim[0] / 2, img_dim[1] / 2
        
        intrinsic = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]).float()
        
        return intrinsic


def scaling_mapping(mapping, new_h, new_w, old_h, old_w):
    """
    Scale mapping coordinates from one resolution to another
    
    Args:
        mapping: Original mapping coordinates
        new_h, new_w: Target resolution
        old_h, old_w: Original resolution
        
    Returns:
        scaled_mapping: Scaled mapping coordinates
    """
    scale_h = new_h / old_h
    scale_w = new_w / old_w
    
    scaled_mapping = mapping.clone()
    scaled_mapping[:, 0] = (mapping[:, 0] * scale_h).long()
    scaled_mapping[:, 1] = (mapping[:, 1] * scale_w).long()
    
    # Clamp to valid range
    scaled_mapping[:, 0] = torch.clamp(scaled_mapping[:, 0], 0, new_h - 1)
    scaled_mapping[:, 1] = torch.clamp(scaled_mapping[:, 1], 0, new_w - 1)
    
    return scaled_mapping
