import torch
import clip
from PIL import Image

class CLIP_OpenAI:
    """
    OpenAI CLIP model wrapper for feature extraction
    """
    
    def __init__(self, cfg):
        self.device = cfg.foundation_model.device
        self.clip_model_name = cfg.foundation_model.clip_model
        
        # Load CLIP model
        self.clip_adapter, self.clip_preprocess = clip.load(
            self.clip_model_name, device=self.device
        )
        
        print(f"Loaded CLIP model: {self.clip_model_name}")
    
    def encode_image(self, images):
        """
        Encode images using CLIP
        
        Args:
            images: Batch of preprocessed images
            
        Returns:
            features: Image features
        """
        with torch.no_grad():
            features = self.clip_adapter.encode_image(images)
        return features
    
    def encode_text(self, texts):
        """
        Encode text using CLIP
        
        Args:
            texts: List of text strings
            
        Returns:
            features: Text features
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            features = self.clip_adapter.encode_text(text_tokens)
        return features
    
    def compute_similarity(self, image_features, text_features):
        """
        Compute cosine similarity between image and text features
        
        Args:
            image_features: Image features
            text_features: Text features
            
        Returns:
            similarity: Similarity scores
        """
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = torch.matmul(image_features, text_features.T)
        return similarity
