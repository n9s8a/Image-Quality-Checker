# Image-Quality-Checker

## Problem Statement
We want to help users **capture and share their best moments**.  

The challenge is to **rank these images by their aesthetic and technical quality**, so that the best, most appealing photos surface first.

## Project Goals
1. **Part 1 – Practical Implementation**
   - Propose and implement a working solution to rank photos by **aesthetic** and **technical quality**.
   - Run directly on the dataset, producing a **ranked CSV file** with per-image scores.
   - Code must be **clean, modular, and extensible**.
---

## Solution Overview

### Features Extracted
- **Aesthetic Score (CLIP)**  
  CLIP similarity with positive and negative prompts.  
  Positive: *"a professional photograph, sharp, vibrant, dramatic lighting"*  
  Negative: *"a blurry, poorly composed snapshot"*  

- **Technical Features**
  - **Sharpness**: Laplacian variance
  - **Exposure**: Mean brightness
  - **Contrast**: Intensity spread
  - **Face Detection**: Presence of faces

- **Composite Scoring**
  - Features are **normalized**.
  - Weighted fusion → `final_score`.
  - Results stored in CSV.

### Evaluation
- **Proxy Metrics** (no labels available):
  - Score standard deviation (spread of rankings)
  - Duplicate fraction
  - Correlation between sharpness and aesthetic score
- **Visualizations**:
  - Score distribution histograms
  - Top-K ranked images

---

## Code Structure

```

image-quality-checker/
│
├── scripts/
│   ├── run_ranking.py         # Main entry point: run pipeline on input folder
│   ├── evaluate.py            # Compute proxy metrics, plots, and top-K visualization
│
├── features/
│   ├── aesthetic.py           # CLIP-based aesthetic scoring with prompts
│   ├── technical.py           # Sharpness, exposure, contrast, face detection
│
├── ranking/
│   ├── fusion.py              # Feature normalization & weighted score fusion
│   ├── dedup.py               # Perceptual hashing for duplicate removal
│
├── evaluation/
│   ├── proxy_metrics.py       # Score_std, duplicate_fraction, sharpness_corr
│   ├── visualization.py       # Histograms + montage of top-K results
│
├── utils/
│   ├── io.py                  # Load/save images, write CSV
│   ├── logging.py             # Centralized logger with timestamps
│
├── config/
│   ├── config.py              # Configurable paths, weights, parameters
│
├── output/
│   ├── csvs/                  # Ranked CSV outputs
│   ├── topk_images/                 # Distribution plots & top-K grids
│
├── results/
├── hyperparameter_tuning.py
├── run_ranking.sh
├── requirements.txt
└── README.md

````
### *Note*: Not using duplicate logic in current implementation. This is for future reference.
---

## Installation

1. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
---

## Usage

### 1. Run Ranking

    ```bash
    python scripts/run_ranking.py \
    --input_dir <path-to-img-dir> \
    --device cpu
    ```

* Extracts features
* Computes composite score
* Saves ranked CSV at:

```
output/csvs/<output>.csv
```

**Sample CSV Output**

| file            | path                      | final_score | aesthetic | sharpness_norm | exposure_norm | contrast_norm | face_present |
| --------------- | ------------------------- | ----------- | --------- | -------------- | ------------- | ------------- | ------------ |
| frame_01050.jpg | .../photos/... | 0.832       | 0.77      | 0.81           | 0.72          | 0.69          | 1            |
| frame_00712.jpg | .../photos/... | 0.791       | 0.74      | 0.78           | 0.66          | 0.71          | 0            |
| ...             | ...                       | ...         | ...       | ...            | ...           | ...           | ...          |

---

### 2. Run Evaluation

```bash
python scripts/evaluate.py --input_csv <path-to-input-csv>
```

**Example Proxy Metrics:**

```
Proxy metrics: {
  'score_std': 0.058,
  'duplicate_fraction': 0.0,
  'sharpness_corr': 0.053
}
```

**Outputs**

* `results/<img_folder_name>.png`
* `output/topk_images/<img_dir>`

---

### 3. Use shall script

```
./run_ranking.sh
```

This will do inference and evaluation together.

## Proxy Metrics Interpretation

* **Score Std**: Spread of ranking scores (higher = better separation).
* **Duplicate Fraction**: Fraction of near-identical images (lower = better).
* **Sharpness Corr**: Correlation between sharpness & aesthetic score (should ideally be positive).

---

## Weight (hyper parameter) tuning:

Weight (Hyperparameter) Tuning

To determine how much importance (weight) each feature should have in the final ranking, we perform hyperparameter tuning.

You can run the tuning script as follows:
```
python hyperparameter_tuning.py
```
This script will search for the best weight values and output the optimal feature weights to use in ranking.

### Best weights we're using:
```
['aesthetic': 0.35,
'sharpness': 0.35,
'exposure': 0.2,
'contrast': 0.1,
'faces': 0.05]
```
