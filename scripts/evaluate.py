import csv, os
import argparse
from config.config import base_csv_dir
from evaluation.proxy_metrics import ProxyEvaluator
from evaluation.visualization import plot_score_distribution, show_topk_images

parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', required=True)
args = parser.parse_args()

input_csv = os.path.join(base_csv_dir, args.input_csv)

ranked_list = []
with open(input_csv, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ranked_list.append({
            'file': row['file'],
            'path': row['path'],
            'final_score': float(row['final_score']),
            'aesthetic': float(row['aesthetic']),
            'sharpness_norm': float(row['sharpness_norm']),
            'exposure_norm': float(row['exposure_norm']),
            'contrast_norm': float(row['contrast_norm']),
            'face_present': int(row['face_present'])
        })

proxy_eval = ProxyEvaluator()
metrics = proxy_eval.evaluate(ranked_list)
print("Proxy metrics:", metrics)

folder_name = os.path.basename(input_csv)
plot_score_distribution(ranked_list, output_path=f'./results/{folder_name}.png')
show_topk_images(ranked_list, k=10, save_dir=f'./output/topk_images/{folder_name}')
