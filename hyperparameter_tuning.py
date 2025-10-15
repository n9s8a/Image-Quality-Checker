import itertools
import os
import tqdm
from features.aesthetic import CLIPAestheticExtractor
from features.technical import TechnicalFeatureExtractor
from ranking.fusion import FeatureFusion
from evaluation.proxy_metrics import ProxyEvaluator
from utils.io import load_images_from_folder
from utils.logging import get_logger

logger = get_logger(__name__)

def optimize_weights_pipeline(input_dir, proxy_eval: ProxyEvaluator,
                              candidate_ranges=None, metric='score_std', device='cpu'):
    """
    Optimize fusion weights for your current ranking pipeline.

    Args:
        input_dir (str): Path to image folder
        proxy_eval (ProxyEvaluator): proxy evaluation instance
        candidate_ranges (dict): weight candidates per feature
        metric (str): metric to maximize ('score_std' or 'sharpness_corr')
        device (str): 'cpu' or 'cuda'

    Returns:
        dict: {'best_weights':..., metric: ...}
    """
    if candidate_ranges is None:
        candidate_ranges = {
            'aesthetic': [0.35, 0.4, 0.45],
            'sharpness': [0.25, 0.3, 0.35],
            'exposure': [0.15, 0.2],
            'contrast': [0.05, 0.1],
            'faces': [0.0, 0.05]
        }

    keys = list(candidate_ranges.keys())
    combos = []
    for vals in itertools.product(*(candidate_ranges[k] for k in keys)):
        if abs(sum(vals) - 1.0) < 1e-6:  # only keep combos summing to 1
            combos.append(dict(zip(keys, vals)))

    logger.info(f"Total weight combinations to evaluate: {len(combos)}")

    aesthetic_extractor = CLIPAestheticExtractor(device=device)
    technical_extractor = TechnicalFeatureExtractor()

    image_paths = load_images_from_folder(input_dir)

    best_metric_val = -float('inf')
    best_weights = None

    for w in combos:
        feature_list = []
        try:
            # Extract features
            for path in image_paths:
                features = {}
                features.update(aesthetic_extractor.extract(path))
                features.update(technical_extractor.extract(path))
                features['path'] = path
                features['file'] = os.path.basename(path)
                feature_list.append(features)

            # Fuse with candidate weights
            fusion = FeatureFusion(weights=w)
            fused_list = fusion.fuse(feature_list)

            # Evaluate proxy metrics
            metrics = proxy_eval.evaluate(fused_list)
            score = metrics.get(metric, -float('inf'))

            if score > best_metric_val:
                best_metric_val = score
                best_weights = w

        except Exception as e:
            logger.exception(f"Failed evaluating weights {w}: {e}")

    logger.info(f"Best weights: {best_weights} -> {metric}={best_metric_val:.4f}")
    return {'best_weights': best_weights, metric: best_metric_val}


# Example usage
if __name__ == "__main__":
    input_dir = "/mnt/c/Users/user/mycode/python/data/<image-dir>/"
    proxy_eval = ProxyEvaluator()
    result = optimize_weights_pipeline(input_dir, proxy_eval, device='cpu')
    print(result)
