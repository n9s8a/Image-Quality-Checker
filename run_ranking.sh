#!/bin/bash
export PYTHONPATH=$(pwd)
python scripts/run_ranking.py \
  --input_dir "<image-dir>" \
  --device "cpu" \
  --topk 100

python scripts/evaluate.py --input_csv "<image-dir>"