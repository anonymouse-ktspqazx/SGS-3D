import torch
import numpy as np

def NMS_cuda(boxes, scores, iou_threshold=0.5):
    """
    CUDA-accelerated Non-Maximum Suppression
    Simplified implementation for anonymous review
    
    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for suppression
        
    Returns:
        keep_indices: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    
    # Sort by scores
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep = []
    while len(sorted_indices) > 0:
        # Take the box with highest score
        current = sorted_indices[0]
        keep.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes[current:current+1]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = compute_box_iou_batch(current_box, remaining_boxes)
        
        # Keep boxes with IoU below threshold
        mask = ious[0] <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]
    
    return torch.stack(keep)

def compute_box_iou_batch(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] boxes in format [x1, y1, x2, y2]
        boxes2: [M, 4] boxes in format [x1, y1, x2, y2]
        
    Returns:
        ious: [N, M] IoU matrix
    """
    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    # Compute union
    union = area1[:, None] + area2 - intersection
    
    # Compute IoU
    ious = intersection / union.clamp(min=1e-6)
    
    return ious
