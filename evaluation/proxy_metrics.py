import numpy as np
from utils.logging import get_logger
from evaluation.base import Evaluator

logger = get_logger(__name__)

class ProxyEvaluator(Evaluator):
    """
    Unsupervised evaluation using proxy metrics:
    - Score distribution
    - Fraction of duplicates
    - Correlation of technical vs aesthetic features
    """

    def evaluate(self, ranked_list):
        """
        Evaluate a ranked list using proxy metrics.

        Args:
            ranked_list (list of dict): ranked images

        Returns:
            dict: {'score_std':..., 'duplicate_fraction':..., 'sharpness_corr':...}
        """
        try:
            scores = np.array([img['final_score'] for img in ranked_list])
            score_std = float(np.std(scores))

            # duplicate fraction
            paths_seen = set()
            duplicates = 0
            for img in ranked_list:
                if img['file'] in paths_seen:
                    duplicates += 1
                else:
                    paths_seen.add(img['file'])
            duplicate_fraction = duplicates / max(len(ranked_list), 1)

            # correlation: sharpness vs aesthetic
            sharpness = np.array([img['sharpness_norm'] for img in ranked_list])
            aesthetic = np.array([img['aesthetic'] for img in ranked_list])
            sharpness_corr = float(np.corrcoef(sharpness, aesthetic)[0,1]) if len(ranked_list)>1 else 0.0

            logger.info(f"Proxy evaluation: score_std={score_std:.4f}, duplicate_fraction={duplicate_fraction:.4f}, sharpness_corr={sharpness_corr:.4f}")
            return {
                'score_std': score_std,
                'duplicate_fraction': duplicate_fraction,
                'sharpness_corr': sharpness_corr
            }
        except Exception as e:
            logger.exception("Proxy evaluation failed")
            return {}
