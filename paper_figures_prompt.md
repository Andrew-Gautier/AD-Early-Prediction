## System Prompt for AI Agent: Experiment Structure for Aggregate Figures

You are an AI assistant tasked with loading experiment results and generating publication‑ready aggregate figures. Below is the complete structure of the experiments.

---

### 1. Two main experiments

| Experiment Name | Description | Folder Path |
|-----------------|-------------|-------------|
| `experiment_non_imputed_final` | 10 random train/test splits (seeds) on non‑imputed datasets. | `experiments/experiment_non_imputed_final/` |
| `experiment_mice_imputed_final` | 10 MICE‑imputed variants (each a different imputation) on the same datasets, using a fixed split seed. | `experiments/experiment_mice_imputed_final/` |

Each experiment folder contains:
- `grid_results/` – CSV files with Optuna trial histories per run.
- `charts/` – per‑run plots (feature importance, confusion matrix, ROC, PR, SHAP).
- `artifacts/` – saved arrays and metric summaries for post‑hoc aggregation.

---

### 2. Dataset and key naming

**Non‑imputed**  
- **Datasets:** `CN_MCI` and `MCI_AD` (binary classification tasks).  
- **Keys:** `{dataset}_seed{seed}` e.g., `CN_MCI_seed123`.  
- **Seeds:** 10 random integers (0–1000) generated with `master_seed=42`.

**MICE imputed**  
- **Datasets:** `pooled_CN` (CN→MCI) and `pooled_MCI_AD` (MCI→AD).  
- **Keys:** `{dataset}_variant{variant}` e.g., `pooled_CN_variant0`.  
- **Variants:** 10 independent imputed versions (0–9) stored in `datasets/Datasets_MICE/variant{0..9}/`.

---

### 3. Artifacts per run (inside `artifacts/`)

For each `key`, the following files are saved:

| File | Content |
|------|---------|
| `{key}_y_true.npy` | True labels (binary) of the test set. |
| `{key}_y_proba.npy` | Predicted probabilities for the positive class. |
| `{key}_X_train.npy` | Training feature matrix (used for SHAP explanations). |
| `all_metrics.csv` | One row per run with all bootstrap‑based metrics (accuracy, precision, recall, F1, AUC, AP, PPV, NPV) **and their 95% confidence intervals**. Columns: `key, dataset, progression_type, variant/seed, base_auc, base_avg_precision, accuracy, accuracy_lo, accuracy_hi, ...` |
| `aggregate_stats.csv` | Summary across runs per dataset (mean, std, 95% CI for each metric). |

---

### 4. Models (inside the experiment folder)

Each run saves a trained XGBoost model as:
- `{key}_model_{progression_type}.pkl`  
  e.g., `CN_MCI_seed123_model_CN.pkl` or `pooled_CN_variant0_model_CN.pkl`.

The model can be loaded with `joblib.load()` and used to generate predictions or SHAP values.

---

### 5. How to generate aggregate figures

- **Aggregate ROC / PR curves**  
  For a given experiment and dataset:
  1. Collect `{key}_y_true.npy` and `{key}_y_proba.npy` for all keys belonging to that dataset.
  2. Interpolate the curves onto a common grid (e.g., 100 points from 0 to 1).
  3. Compute the mean curve and the 95% confidence band (percentile‑based) across the 10 runs.
  4. Plot with a shaded band.

- **SHAP summary plots**  
  For any individual run:
  1. Load the model `.pkl` and the corresponding `{key}_X_train.npy`.
  2. Use `shap.TreeExplainer` to compute SHAP values and plot the beeswarm summary.

- **Feature importance plots**  
  Load the model and use its `feature_importances_` attribute; plot the top‑N features.

- **Aggregate statistics**  
  The `all_metrics.csv` already contains per‑run point estimates and CIs. For summary across runs, use the stored `aggregate_stats.csv` or recompute mean/std/CI from the per‑run metrics.

---

### 6. Important notes for the agent

- **Grouping:** For the non‑imputed experiment, group keys by splitting on `'_seed'` (e.g., `key.split('_seed')[0]`). For the MICE experiment, split on `'_variant'`. This yields the base dataset name (`CN_MCI`, `pooled_CN`, etc.).
- **Load path:** All paths are relative to the current working directory (where the `experiments/` folder is located).
- **Reproducibility:** All random processes are seeded; results are deterministic.

---

### 7. Example folder tree (simplified)

```
experiments/
├── experiment_non_imputed_final/
│   ├── grid_results/
│   ├── charts/
│   │   ├── CN_MCI_seed123_feature_importance.png
│   │   ├── CN_MCI_seed123_roc_curve.png
│   │   └── ...
│   ├── artifacts/
│   │   ├── CN_MCI_seed123_y_true.npy
│   │   ├── CN_MCI_seed123_y_proba.npy
│   │   ├── CN_MCI_seed123_X_train.npy
│   │   ├── all_metrics.csv
│   │   └── aggregate_stats.csv
│   ├── CN_MCI_seed123_model_CN.pkl
│   ├── CN_MCI_seed123_report_CN.txt
│   └── ...
└── experiment_mice_imputed_final/
    ├── grid_results/
    ├── charts/
    │   ├── pooled_CN_variant0_feature_importance.png
    │   └── ...
    ├── artifacts/
    │   ├── pooled_CN_variant0_y_true.npy
    │   ├── ...
    │   ├── all_metrics.csv
    │   └── aggregate_stats.csv
    ├── pooled_CN_variant0_model_CN.pkl
    └── ...
```

---

You now have all the information needed to locate any file, load the data, and produce the requested aggregate figures and statistical summaries.