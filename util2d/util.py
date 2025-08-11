import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils

def show_mask(mask, ax, random_color=False):
    """
    Visualize a mask on matplotlib axis
    
    Args:
        mask: Binary mask array
        ax: Matplotlib axis
        random_color: Whether to use random color
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def masks_to_rle(masks):
    """
    Convert masks to RLE format for efficient storage
    
    Args:
        masks: Tensor of masks [N, H, W]
        
    Returns:
        rles: List of RLE encoded masks
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    rles = []
    for mask in masks:
        if len(mask.shape) == 3:
            mask = mask[0]  # Remove channel dimension if present
        
        # Convert to uint8
        mask = (mask > 0).astype(np.uint8)
        
        # Encode to RLE
        rle = mask_utils.encode(np.asfortranarray(mask))
        rles.append(rle)
    
    return rles

def rle_to_masks(rles, height, width):
    """
    Convert RLE format back to masks
    
    Args:
        rles: List of RLE encoded masks
        height: Image height
        width: Image width
        
    Returns:
        masks: Tensor of masks [N, H, W]
    """
    masks = []
    for rle in rles:
        mask = mask_utils.decode(rle)
        masks.append(torch.from_numpy(mask))
    
    if masks:
        return torch.stack(masks)
    else:
        return torch.empty(0, height, width)

def filter_masks_by_area(masks, min_area=100, max_area_ratio=0.9):
    """
    Filter masks by area constraints
    
    Args:
        masks: Tensor of masks [N, H, W]
        min_area: Minimum mask area in pixels
        max_area_ratio: Maximum mask area as ratio of image area
        
    Returns:
        filtered_masks: Filtered masks
        valid_indices: Indices of valid masks
    """
    if len(masks) == 0:
        return masks, []
    
    h, w = masks.shape[-2:]
    total_area = h * w
    max_area = total_area * max_area_ratio
    
    areas = torch.sum(masks.view(len(masks), -1), dim=1)
    valid_mask = (areas >= min_area) & (areas <= max_area)
    valid_indices = torch.where(valid_mask)[0]
    
    return masks[valid_indices], valid_indices.tolist()

def compute_mask_iou(mask1, mask2):
    """
    Compute IoU between two masks
    
    Args:
        mask1, mask2: Binary masks
        
    Returns:
        iou: Intersection over Union
    """
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def compute_mask_overlap_matrix(masks):
    """
    Compute pairwise IoU matrix for a set of masks
    
    Args:
        masks: Tensor of masks [N, H, W]
        
    Returns:
        iou_matrix: NxN matrix of pairwise IoUs
    """
    n_masks = len(masks)
    iou_matrix = np.zeros((n_masks, n_masks))
    
    for i in range(n_masks):
        for j in range(i, n_masks):
            if i == j:
                iou_matrix[i, j] = 1.0
            else:
                iou = compute_mask_iou(masks[i], masks[j])
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
    
    return iou_matrix

def non_maximum_suppression_masks(masks, scores, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to masks based on IoU
    
    Args:
        masks: Tensor of masks [N, H, W]
        scores: Confidence scores for each mask
        iou_threshold: IoU threshold for suppression
        
    Returns:
        keep_indices: Indices of masks to keep
    """
    if len(masks) == 0:
        return []
    
    # Sort by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep_indices = []
    
    for i in sorted_indices:
        # Check if this mask overlaps significantly with any kept mask
        should_keep = True
        
        for kept_idx in keep_indices:
            iou = compute_mask_iou(masks[i], masks[kept_idx])
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep_indices.append(i.item())
    
    return keep_indices

def visualize_masks_on_image(image, masks, alpha=0.5, colors=None):
    """
    Visualize masks overlaid on an image
    
    Args:
        image: RGB image array [H, W, 3]
        masks: Tensor of masks [N, H, W]
        alpha: Transparency for mask overlay
        colors: Optional list of colors for each mask
        
    Returns:
        vis_image: Image with mask overlay
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    vis_image = image.copy()
    
    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        if colors is not None and i < len(colors):
            color = colors[i]
        else:
            color = np.random.randint(0, 255, 3)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 1-alpha, colored_mask, alpha, 0)
    
    return vis_image

def save_masks_as_images(masks, output_dir, prefix="mask"):
    """
    Save individual masks as image files
    
    Args:
        masks: Tensor of masks [N, H, W]
        output_dir: Output directory
        prefix: Filename prefix
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Convert to 0-255 range
        mask_img = (mask * 255).astype(np.uint8)
        
        filename = f"{prefix}_{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, mask_img)

def load_masks_from_images(input_dir, prefix="mask"):
    """
    Load masks from image files
    
    Args:
        input_dir: Input directory
        prefix: Filename prefix
        
    Returns:
        masks: Tensor of loaded masks
    """
    import os
    import glob
    
    pattern = os.path.join(input_dir, f"{prefix}_*.png")
    mask_files = sorted(glob.glob(pattern))
    
    masks = []
    for mask_file in mask_files:
        mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = torch.from_numpy(mask_img > 0).float()
        masks.append(mask)
    
    if masks:
        return torch.stack(masks)
    else:
        return torch.empty(0, 0, 0)
