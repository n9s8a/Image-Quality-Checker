import numpy as np
from utils.logging import get_logger

logger = get_logger(__name__)

class FeatureFusion:
    """
    Combines multiple feature dicts into a single final score using weighted sum.
    """

    def __init__(self, weights=None):
        """
        Args:
            weights (dict): optional custom weights per feature
        """
        self.weights = weights or {
            'aesthetic': 0.35,
            'sharpness': 0.35,
            'exposure': 0.2,
            'contrast': 0.1,
            'faces': 0.05
        }

    def normalize(self, values, log_scale=False):
        """
        Min-max normalize an array to 0..1
        """
        try:
            values = np.array(values)
            if log_scale:
                values = np.log1p(values)
            normed = (values - values.min()) / (np.ptp(values)+1e-9)
            return normed
        except Exception as e:
            logger.exception("Normalization failed")
            raise RuntimeError(f"Normalization failed: {e}")

    def fuse(self, feature_dicts):
        """
        Args:
            feature_dicts (list of dicts): each dict contains features for one image

        Returns:
            list of dicts: each dict includes 'final_score'
        """
        try:
            for key in ['sharpness','exposure','contrast']:
                vals = [fd[key] for fd in feature_dicts]
                normed = self.normalize(vals, log_scale=(key=='sharpness'))
                for i, fd in enumerate(feature_dicts):
                    fd[f"{key}_norm"] = float(normed[i])

            for fd in feature_dicts:
                fd['face_present'] = int(fd['faces']>0)

            for fd in feature_dicts:
                score = (
                    self.weights.get('aesthetic',0)*fd.get('aesthetic',0) +
                    self.weights.get('sharpness',0)*fd.get('sharpness_norm',0) +
                    self.weights.get('exposure',0)*fd.get('exposure_norm',0) +
                    self.weights.get('contrast',0)*fd.get('contrast_norm',0) +
                    self.weights.get('faces',0)*fd.get('face_present',0)
                )
                fd['final_score'] = float(score)
            logger.info("Feature fusion completed")
            return feature_dicts
        except Exception as e:
            logger.exception("Feature fusion failed")
            raise RuntimeError(f"Feature fusion failed: {e}")
