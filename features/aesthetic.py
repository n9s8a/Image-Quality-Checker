import clip
import torch
import math
from config.config import prompts
from .base import FeatureExtractor
from utils.io import load_image_pil
from utils.logging import get_logger

logger = get_logger(__name__)

class CLIPAestheticExtractor(FeatureExtractor):
    """
    Extracts aesthetic scores for images using CLIP embeddings.
    Computes similarity to positive/negative text prompts.

    Returns:
        dict: {'aesthetic': float in 0..1}
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.model, self.preprocess = clip.load('ViT-B/32', device=device)
        self.pos_prompts = prompts['positive']
        self.neg_prompts = prompts['negative']

        self._init_text_features()

    def _init_text_features(self):
        """Initialize and normalize text embeddings for positive/negative prompts."""
        try:
            pos_tokens = clip.tokenize(self.pos_prompts).to(self.device)
            neg_tokens = clip.tokenize(self.neg_prompts).to(self.device)
            with torch.no_grad():
                pos_feats = self.model.encode_text(pos_tokens).float()
                neg_feats = self.model.encode_text(neg_tokens).float()
                self.pos_text = pos_feats.mean(dim=0, keepdim=True)
                self.neg_text = neg_feats.mean(dim=0, keepdim=True)
                self.pos_text /= self.pos_text.norm(dim=-1, keepdim=True)
                self.neg_text /= self.neg_text.norm(dim=-1, keepdim=True)
            logger.info("CLIP text features initialized.")
        except Exception as e:
            logger.exception("Failed to initialize CLIP text features.")
            raise RuntimeError(f"Failed to initialize CLIP text features: {e}")

    def extract(self, image_path):
        """
        Compute aesthetic score for a single image.

        Args:
            image_path (str): Path to the image.

        Returns:
            dict: {'aesthetic': float}

        Raises:
            RuntimeError on failure.
        """
        try:
            pil_image = load_image_pil(image_path)
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = self.model.encode_image(image_tensor).float()
                img_feat /= img_feat.norm(dim=-1, keepdim=True)
                pos_sim = (img_feat @ self.pos_text.T).squeeze().item()
                neg_sim = (img_feat @ self.neg_text.T).squeeze().item()
                score = pos_sim - neg_sim
                mapped = 1 / (1 + math.exp(-5 * score))
                return {'aesthetic': float(mapped)}
        except Exception as e:
            logger.exception(f"Failed to extract aesthetic for {image_path}")
            raise RuntimeError(f"Failed to extract CLIP aesthetic for {image_path}: {e}")
