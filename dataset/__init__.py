"""
Anonymous 3D Instance Segmentation - Dataset Module

This module contains simplified dataset loaders for anonymous review.
Full implementation will be released upon paper acceptance.
"""

from .scannet_loader import ScanNetReader, scaling_mapping

def build_dataset(root_path, cfg):
    """
    Build dataset loader based on configuration
    
    Args:
        root_path: Path to dataset
        cfg: Configuration object
        
    Returns:
        loader: Dataset loader instance
    """
    dataset_name = cfg.data.dataset_name
    
    if 'scannet' in dataset_name.lower():
        return ScanNetReader(root_path=root_path, cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

__all__ = ['build_dataset', 'ScanNetReader', 'scaling_mapping']
