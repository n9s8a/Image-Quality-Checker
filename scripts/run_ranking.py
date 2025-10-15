import argparse, os
from config.config import base_input_dir
from features.aesthetic import CLIPAestheticExtractor
from features.technical import TechnicalFeatureExtractor
from ranking.fusion import FeatureFusion
from ranking.dedup import Deduplicator
from utils.io import load_images_from_folder, save_csv
from utils.logging import get_logger
import tqdm

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output', default='ranked.csv')
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    input_dir = os.path.join(base_input_dir, args.input_dir)
    folder_name = os.path.basename(args.input_dir)
    output_csv = f"./output/csvs/{folder_name}.csv"

    logger.info(f"Starting ranking on {args.input_dir}")

    aesthetic_extractor = CLIPAestheticExtractor(device=args.device)
    technical_extractor = TechnicalFeatureExtractor()
    fusion = FeatureFusion()
    dedup = Deduplicator()

    image_paths = load_images_from_folder(input_dir)
    feature_list = []

    for path in tqdm.tqdm(image_paths, desc="Extracting features"):
        try:
            features = {}
            features.update(aesthetic_extractor.extract(path))
            features.update(technical_extractor.extract(path))
            features['path'] = path
            features['file'] = path.split('/')[-1]
            feature_list.append(features)
        except Exception as e:
            logger.error(f"Feature extraction failed for {path}: {e}")

    fused = fusion.fuse(feature_list)

    # deduped = dedup.dedup(fused)
    ranked = sorted(fused, key=lambda x: x['final_score'], reverse=True)

    fieldnames = ['file','path','final_score','aesthetic','sharpness_norm','exposure_norm','contrast_norm','face_present']
    save_csv(output_csv, ranked, fieldnames)
    logger.info(f"Saved ranking CSV to {args.output}")

    print(f"Top {args.topk} images:")
    for i, r in enumerate(ranked[:args.topk],1):
        print(f"{i:03d}. {r['file']}  score={r['final_score']:.4f} aest={r['aesthetic']:.3f} sharp={r['sharpness_norm']:.3f} exp={r['exposure_norm']:.3f} ctr={r['contrast_norm']:.3f} faces={r['face_present']}")

if __name__ == '__main__':
    main()
