import torch
from segment_anything import sam_model_registry, SamPredictor

class SAM_HQ:
    """
    Segment Anything Model wrapper
    """
    
    def __init__(self, cfg):
        self.device = cfg.foundation_model.device
        self.checkpoint_path = cfg.foundation_model.sam_checkpoint
        
        # Load SAM model
        self.sam_model = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
        self.sam_model.to(device=self.device)
        
        # Create predictor
        self.sam_predictor = SamPredictor(self.sam_model)
        
        print("Loaded SAM model")
    
    def predict_masks(self, image, boxes=None, points=None, point_labels=None):
        """
        Predict masks using SAM
        
        Args:
            image: Input image
            boxes: Bounding boxes for prompts
            points: Point prompts
            point_labels: Labels for point prompts
            
        Returns:
            masks: Predicted masks
            scores: Confidence scores
            logits: Raw logits
        """
        self.sam_predictor.set_image(image)
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=boxes,
            multimask_output=False
        )
        
        return masks, scores, logits
