# Sports Photo Quality Ranking

## Problem Statement
We want to help recreational players **capture and share their best sporting moments**.  

From 10-minute sports match videos, ~100 frames are automatically extracted that show **active gameplay** (e.g., hitting a shot, team mid-rally).  

The next challenge is to **rank these images by their aesthetic and technical quality**, so that the best, most appealing photos surface first.

## Project Goals
1. **Part 1 – Practical Implementation**
   - Propose and implement a working solution to rank sports photos by **aesthetic** and **technical quality**.
   - Run directly on the dataset, producing a **ranked CSV file** with per-image scores.
   - Code must be **clean, modular, and extensible**.

2. **Part 2 – Long-term Ideal Solution**
   - Discuss how to extend this into a **production-grade ranking system**, including:
     - Data collection & labeling
     - Model training
     - Evaluation metrics
     - Deployment strategy

---

## Solution Overview

### Features Extracted
- **Aesthetic Score (CLIP)**  
  CLIP similarity with positive and negative prompts.  
  Positive: *"a professional sports photograph, sharp, vibrant, dramatic lighting"*  
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

sports-photo-quality-ranking/
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

1. Clone repository
   ```bash
   git clone https://github.com/n9s8a/sports-photo-quality-ranking.git
   cd sports-photo-quality-ranking
    ````

2. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```
---

## Usage

### 1. Run Ranking

    ```bash
    python scripts/run_ranking.py \
    --input_dir photos_assignment/2025_05_08_match_dir \
    --device cpu
    ```

* Extracts features
* Computes composite score
* Saves ranked CSV at:

```
output/csvs/2025_05_08_match_dir.csv
```

**Sample CSV Output**

| file            | path                      | final_score | aesthetic | sharpness_norm | exposure_norm | contrast_norm | face_present |
| --------------- | ------------------------- | ----------- | --------- | -------------- | ------------- | ------------- | ------------ |
| frame_01050.jpg | .../photos_assignment/... | 0.832       | 0.77      | 0.81           | 0.72          | 0.69          | 1            |
| frame_00712.jpg | .../photos_assignment/... | 0.791       | 0.74      | 0.78           | 0.66          | 0.71          | 0            |
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

# Part 2: Long-term Ideal Solution
## Data Collection & Labeling

Gather a larger set of images from different sports, lighting conditions, and camera types to make the model more general.

Instead of asking people to give absolute scores, use pairwise comparisons: show two images and ask “Which one would you share or post?” – this gives more reliable ranking information.

Collect additional labels for technical quality like sharpness, exposure, motion blur, and composition.

Ensure quality control by including known “gold standard” image pairs, checking agreement between multiple raters, and having multiple people label the same pairs.

Optionally, use active learning to focus labeling on the most informative image pairs, which reduces effort while improving the model.

## Model Training

Fine-tune pre-trained models like CLIP or NIMA using the images and human-provided labels. This helps the model learn what makes an image look good.

Predict both visual appeal and technical quality (like sharpness, brightness, and contrast) so the ranking considers all important aspects.

Optionally, combine different features—for example, the model’s embeddings from CLIP and the technical measurements—using simple models like a small neural network or LightGBM. This can make the ranking more reliable.

If your labels are pairwise comparisons (“which image looks better?”), train the model to focus on relative rankings instead of absolute scores. This helps it better mimic human preferences.

## Evaluation Metrics

Ranking correlation – Check how well the model’s ranking matches human judgment. We can use methods like Spearman’s rho or Kendall’s tau to see if the top images chosen by the model are the same ones humans prefer.

Top-K relevance – Focus on the top few images the model ranks as best. Metrics like NDCG@K tell us how many of these top images are actually high-quality according to humans.

Human evaluation – Show people the top-ranked images in A/B tests (side-by-side comparisons) to see if the model’s choices are truly appealing in real-world scenarios.

## Deployment

Expose ranking as a FastAPI service that receives images or precomputed features and returns ranked scores (example).

Cache CLIP embeddings and technical features for efficient batch inference.

Store ranked results with metadata in a database for downstream use or analytics.

Monitor score distribution drift, user engagement, and periodically update the model using new labeled data or active learning.

Integrate ranking into the video-to-photo extraction pipeline so high-quality images are surfaced automatically for users.