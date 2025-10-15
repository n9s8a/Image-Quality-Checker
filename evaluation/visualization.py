import matplotlib.pyplot as plt
from utils.logging import get_logger
import os
from PIL import Image

logger = get_logger(__name__)

def plot_score_distribution(ranked_list, output_path='./results/score_hist.png'):
    """
    Plot histogram of final scores
    """
    try:
        scores = [img['final_score'] for img in ranked_list]
        plt.figure(figsize=(6,4))
        plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Final Score')
        plt.ylabel('Number of Images')
        plt.title('Distribution of Image Scores')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Score distribution saved to {output_path}")
    except Exception as e:
        logger.exception("Failed to plot score distribution")

def show_topk_images(ranked_list, k=5, save_dir='./output/topk_images'):
    """
    Save top-K images to a folder for visual inspection
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(ranked_list[:k],1):
            pil_img = Image.open(img['path']).convert('RGB')
            save_path = os.path.join(save_dir, f"{i:02d}_{img['file']}")
            pil_img.save(save_path)
        logger.info(f"Saved top-{k} images to {save_dir}")
    except Exception as e:
        logger.exception("Failed to save top-K images")
