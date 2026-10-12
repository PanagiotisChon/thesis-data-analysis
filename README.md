# 🏠 Residential Property Valuation Using Machine Learning Techniques

Bachelor's thesis — **University of Piraeus**, School of Information and Communication Technologies, Department of Digital Systems.

- **Student:** Panagiotis Chondrompilas
- **Supervisor:** Michail Filippakis, Professor
- **Date:** 29 May 2026

This repository contains the complete code (pipeline) and the final text of the thesis, which investigates the prediction of residential sale prices in the Greek real estate market (Attica and Thessaloniki) using **regression** (supervised learning) and **clustering** (unsupervised learning) algorithms.

---

## 📑 Table of Contents

1. [Summary](#-summary)
2. [Repository Structure](#-repository-structure)
3. [Dataset](#-dataset)
4. [Methodology](#-methodology)
5. [Results](#-results)
6. [Installation and Usage](#-installation-and-usage)
7. [Output Files](#-output-files)
8. [Limitations](#-limitations)
9. [Future Work](#-future-work)
10. [License and Citation](#-license-and-citation)

---

## 🔎 Summary

The thesis follows an end-to-end machine learning workflow:

1. Exploratory data analysis (EDA): missing values, outliers (IQR rule), descriptive statistics.
2. Cleaning with **domain-driven thresholds** rather than purely statistical rules.
3. Feature engineering: **target encoding with smoothing** for the neighborhood, one-hot encoding, standardization.
4. **Logarithmic transformation** of the target variable (`log1p`) for the tree-based models.
5. Training and comparison of **4 regression models**: Linear Regression, Decision Tree, Random Forest, HistGradientBoosting.
6. **Clustering** with K-Means and DBSCAN, visualized with PCA.

**Key finding:** tree-based ensemble models (Random Forest, HistGradientBoosting) reach **R² ≈ 0.72** and **MAPE ≈ 25%**, clearly outperforming linear regression (R² 0.64, MAPE 43%). Property area, neighborhood and construction year account for roughly 87% of the Random Forest's predictive power.

---

## 📁 Repository Structure

```
.
├── final_pipeline_updated.py     # Full pipeline: EDA → preprocessing → regression → clustering
├── Project_Documentation.pdf     # Final text of the thesis (in Greek)
├── greecelistings.csv            # Dataset (download from Kaggle — see below)
├── requirements.txt              # Python dependencies                
└── README.md                     # README
```

> Plots and result files (`*.png`, `*.csv`, `final_summary.json`) are generated automatically by the script in the working directory.

---

## 📊 Dataset

The public **"Greece Listings"** dataset from [Kaggle](https://www.kaggle.com/) is used (residential sale listings).

| Property | Value |
| --- | --- |
| Observations (initial) | 20,000 |
| Columns | 25 |
| Observations after cleaning | 19,518 (482 removed, 2.41%) |
| Geographic distribution | Attica 94.06% · Thessaloniki 5.94% |
| Target variable | `res_price` (asking price in €) |

**Distribution by property type:**

| Type | Count | Share | Median price |
| --- | ---: | ---: | ---: |
| Apartment | 13,899 | 69.50% | €170,000 |
| Detached house | 2,563 | 12.82% | €370,000 |
| Maisonette | 2,511 | 12.56% | €450,000 |
| Building | 880 | 4.40% | €600,000 |
| House (residence) | 147 | 0.74% | €360,000 |

**Main variables:** `res_sqr` (area), `construction_year`, `bedrooms`, `bathrooms`, `levels`, `location_region`, `location_name` (neighborhood), `status`, `energyclass`, `parking`, and binary amenities (`auto_heating`, `solar`, `cooling`, `safe_door`, `gas`, `fireplace`, `furniture`, `student`).

> ⚠️ The file `greecelistings.csv` is included in the repository and the script reads it from the same folder. The original source of the dataset is Kaggle (see above).
---

## ⚙️ Methodology

### 1. Outlier removal

Logical thresholds are applied so that legitimately expensive properties are not removed:

| Variable | Accepted range |
| --- | --- |
| `res_price` | €20,000 – €3,000,000 |
| `res_price_sqr` | €300 – €12,000 per sq.m. |
| `res_sqr` | 20 – 1,000 sq.m. |
| `bedrooms` | 1 – 8 |
| `bathrooms` | 1 – 6 |
| `construction_year` | 1900 – 2025 |

### 2. Missing value handling

| Variable | Strategy |
| --- | --- |
| `parking` | Missing is interpreted as "no parking" → binary `has_parking` |
| `energyclass`, `status`, `location_name` | Filled with the category "Άγνωστο" (Unknown) |
| `bedrooms`, `bathrooms`, `construction_year` | **Per-property-type** median (group-wise median imputation) |
| `levels` | Only the first listed floor is kept (reduces the 110 original categories) |
| `res_date`, `res_address`, `deleted`, `deleted_at` | Dropped (non-predictive) |

### 3. Feature engineering

- **Avoiding data leakage:** `res_price_sqr` is removed from the features, since it is derived directly from the target price.
- **Target encoding of the neighborhood with smoothing** (`k = 20`), computed **on the training set only**:

  ```
  enc = (n · μ_neighborhood + k · μ_global) / (n + k)
  ```

  A `loc_frequency` feature (how often the neighborhood appears) is also created. Neighborhoods unseen in the test set receive the global mean.
- **One-Hot Encoding** (`drop_first=True`) for: `location_region`, `res_type`, `levels`, `status`, `energyclass`.
- **StandardScaler** on the numeric features: `res_sqr`, `construction_year`, `bedrooms`, `bathrooms`, `loc_target_enc`, `loc_frequency` (fit on the training set only).
- **Train/test split:** 80/20, `random_state=42` (test set = 3,904 observations).
- **Log-transform** of the target (`np.log1p`) for Decision Tree, Random Forest and HistGradientBoosting; predictions are converted back with `np.expm1`.

### 4. Regression models

| Model | Key hyperparameters | Target |
| --- | --- | --- |
| Linear Regression | defaults | `price` (no log) |
| Decision Tree | `max_depth=12`, `min_samples_leaf=5` | `log1p(price)` |
| Random Forest | `n_estimators=200`, `max_depth=20`, `min_samples_leaf=2` | `log1p(price)` |
| HistGradientBoosting | `max_iter=500`, `max_depth=8`, `learning_rate=0.05`, `min_samples_leaf=20` | `log1p(price)` |

**Evaluation metrics:** R², MSE, RMSE, MAE, MAPE.

### 5. Clustering

- **Features:** `res_sqr`, `construction_year`, `bedrooms`, `bathrooms`, `has_parking`, `fireplace`, `solar`, `cooling`, neighborhood mean price (`loc_mean_price`).
- **K-Means:** `k` selected (2–8) using the elbow method and silhouette score (on a sample of up to 5,000 points).
- **DBSCAN:** `eps=0.8`, `min_samples=20`.
- **Visualization:** PCA to 2 dimensions.

---

## 📈 Results

### Regression (test set)

| Model | R² | RMSE (€) | MAE (€) | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 0.6374 | 209,900 | 115,215 | 43.26 |
| Decision Tree | 0.6527 | 205,418 | 98,843 | 30.96 |
| **Random Forest** | **0.7226** | **183,572** | **83,478** | 25.34 |
| HistGradientBoosting | 0.7201 | 184,402 | 85,243 | **24.98** |

Random Forest has the best R² and RMSE, while HistGradientBoosting has the lowest MAPE. In practice the two are equivalent, with HistGradientBoosting training considerably faster (~4 s vs ~17 s).

### Feature importance (Random Forest)

| # | Feature | Importance |
| ---: | --- | ---: |
| 1 | `res_sqr` (area) | 66.5% |
| 2 | `loc_target_enc` (neighborhood mean price) | 10.3% |
| 3 | `construction_year` | 10.0% |
| 4 | `loc_frequency` | 2.0% |
| 5 | `safe_door` | 1.1% |
| 6 | `bedrooms` | 0.7% |
| 7 | `levels_Ισόγειο` (ground floor) | 0.5% |
| 8 | `fireplace` | 0.5% |

### Clustering

| Algorithm | Parameters | Clusters | Noise | Silhouette |
| --- | --- | ---: | ---: | ---: |
| K-Means | k = 2 | 2 | — | 0.2760 |
| DBSCAN | eps = 0.8, min_samples = 20 | 58 | 5,355 (27.4%) | 0.2122 |

PCA (2 components) retains 50.5% of the variance.

**Profile of the 2 K-Means clusters:**

| Cluster | n | Mean price | Median price | Area | Bedrooms | Year | Parking | Fireplace |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1 — Premium** | 7,478 | €560,423 | €410,000 | 206 sq.m. | 3.4 | 2004 | 34% | 76% |
| **0 — Budget** | 12,040 | €175,862 | €145,000 | 89 sq.m. | 2.0 | 1982 | 6% | 5% |

K-Means yields two interpretable categories (larger, newer premium properties versus smaller, older budget apartments), while DBSCAN is found unsuitable for this dataset.

### Practical interpretation

A MAPE of ≈ 25% means that for a property worth €200,000 the prediction deviates by about €50,000 on average. This is acceptable for **indicative estimates and initial screening of listings**, but not for official bank valuations or notarial deeds.

---

## 🚀 Installation and Usage

### Prerequisites

- Python 3.9 or newer

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/PanagiotisChon/thesis-data-analysis.git
cd thesis-data-analysis

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place greecelistings.csv (from Kaggle) in the same folder

# 5. Run
python final_pipeline_updated.py
```

### `requirements.txt`

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Running the pipeline prints statistics, metrics and cluster profiles to the console, and saves all plots and result files. The random seed is fixed (`42`) for reproducible results.

> **Note:** plot titles and labels are in Greek.

---

## 🗂️ Output Files

| File | Description |
| --- | --- |
| `boxplots_raw.png` | Boxplots before cleaning |
| `boxplots_clean.png` | Boxplots after cleaning |
| `histograms.png` | Histograms of numeric features |
| `heatmap.png` | Pearson correlation heatmap |
| `pairplot.png` | Pairplot (price, area, bedrooms, bathrooms) |
| `predicted_vs_actual_v2.png` | Predicted vs actual prices (4 models) |
| `error_distribution_v2.png` | Error distribution per model |
| `model_comparison_v2.png` | R² and RMSE comparison |
| `feature_importance_v2.png` | Top-15 features (Random Forest) |
| `feature_importance_final.csv` | Full feature importance table |
| `kmeans_selection_v2.png` | Elbow method and silhouette score |
| `kmeans_profile_v2.csv` | Cluster profiles in real units |
| `clusters_pca_v2.png` | K-Means and DBSCAN visualized in 2D PCA |
| `final_summary.json` | Summary of results (metrics, clustering, PCA) |

---

## ⚠️ Limitations

- **Geographic imbalance:** 94% Attica, 6% Thessaloniki; results do not generalize to islands or provincial cities.
- **Asking prices, not transaction prices:** asking prices are typically 5–15% higher than final sale prices, so the model tends to overestimate.
- **No temporal dimension:** `res_date` was dropped, so price evolution over time is not captured.
- **Limited geographic precision:** the neighborhood is a coarse proxy; there are no coordinates, nor information on view, orientation or proximity to public transport.
- **Underestimation of expensive properties:** all models underestimate properties above ~€1.5 million.
- **No hyperparameter tuning:** hyperparameters are reasonable but not optimized.
- **Moderate silhouette in clustering:** properties vary continuously and do not split into "natural" clusters.

---

## 🔮 Future Work

- Enrichment with **geographic coordinates** and distance to metro stations, the sea, the city center.
- Use of the **temporal dimension** (listing date) and **macroeconomic indicators** (interest rates, inflation).
- Use of **actual transaction prices** instead of asking prices.
- Systematic **hyperparameter tuning** (Grid Search / Bayesian optimization) and cross-validation.
- Testing XGBoost / LightGBM and interpretability techniques (SHAP).
- Extension to more regions of Greece.

---

## 📜 License and Citation

The code and text are provided for academic and educational purposes. The "Greece Listings" dataset is subject to the terms of use of its original author on Kaggle.

If you use this work, please cite:

```
Chondrompilas, P. (2026). Residential Property Valuation Using Machine Learning Techniques.
Bachelor's Thesis, University of Piraeus, Department of Digital Systems.
```
