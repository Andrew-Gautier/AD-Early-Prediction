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

### 8. Lead-time analysis experiment (clinical early-detection results)

In addition to the two training experiments above, there is a separate **lead-time analysis** that measures how early each trained model can detect progression (CN→MCI or MCI→AD) before the true conversion visit, and how often it falsely alarms on non-progressors (controls).

**Key files/folders (all at the workspace root, `AD-Early-Prediction/`):**

| Path | Content |
|------|---------|
| `leadtime.py` | Shared module with all lead-time analysis functions (see below). |
| `leadtime_analysis.ipynb` | Notebook that runs the full grid-search/aggregation and produces the figures below. |
| `lead_time results/` | Output folder for the notebook's aggregated CSVs and PNG charts. |
| `leadtime_prob_cache/` | Cached per-model prediction-probability CSVs, used to avoid re-fitting/re-scoring models when re-running the analysis at different thresholds/mask lengths. |
| `datasets/Dataset_v2_1/lead_time_CN.csv`, `lead_time_MCI_AD.csv` | Source data (with `n_visits`, `Prog_ID`, `Progression`, `months_since_baseline`, etc.) fed into `run_leadtime()`. |

**`leadtime_prob_cache/` structure:**

```
leadtime_prob_cache/
├── non‑imputed/                       # NOTE: uses a figure-dash "‑" (U+2011), not a regular hyphen "-"
│   ├── CN_MCI/
│   │   ├── CN_MCI_seed104/
│   │   │   ├── lead_time_probabilities.csv
│   │   │   ├── lead_time_predictions.csv
│   │   │   ├── control_probabilities.csv
│   │   │   └── control_predictions.csv
│   │   └── ... (one folder per seed, 10 total: seed104, seed114, seed142, seed228, seed25, seed250, seed281, seed654, seed754, seed759)
│   └── MCI_AD/
│       └── MCI_AD_seed{...}/ (same 4 files, same 10 seeds)
└── MICE‑imputed/
    ├── CN_MCI/
    │   └── pooled_CN_variant{0..9}/ (same 4 files, 10 variants)
    └── MCI_AD/
        └── pooled_MCI_AD_variant{0..9}/ (same 4 files, 10 variants)
```

- **Model key naming** mirrors the training experiments: `{dataset}_seed{seed}` for non-imputed (e.g. `CN_MCI_seed104`), `{dataset}_variant{n}` for MICE-imputed (e.g. `pooled_CN_variant0`).
- **Cohort folder** (`CN_MCI` or `MCI_AD`) always matches the progression task, regardless of imputation type or dataset key prefix (`pooled_CN` models live under `CN_MCI`, `pooled_MCI_AD` models live under `MCI_AD`).
- **`lead_time_probabilities.csv` / `lead_time_probabilities.csv` schema:** columns `ID`, `months_since_baseline` (stringified list, parse with `_parse_list`/`ast.literal_eval`), `Progression` (stringified tuple of stage per visit), then one numbered column per truncation (`'0'`, `'1'`, `'2'`, ...) holding the model's predicted probability (or 0/1 prediction, for the `_predictions.csv` files) at that truncation. Truncation `i` uses the first `i+2` visits. `control_*` files have the same schema but for non-progressors, and were generated with a given `mask_length` (visits held out from the end) at cache-build time — masking for progressors is instead applied post-hoc when scoring (see `analyze_run` below).

**Key functions in `leadtime.py`:**

| Function | Purpose |
|----------|---------|
| `run_leadtime(file_path, dest_dir, model, progression_type, mask_length=0)` | Runs one model over every truncation of every subject, writing the 4 CSVs above to `dest_dir`. |
| `load_models(experiment_dir, progression_type)` | Loads all `*_model_{CN,AD}.pkl` files from an experiment folder into a `{key: model}` dict. |
| `get_probabilities(models_dict, leadtime_csv, prog_type, cache_dir)` | For each model, loads cached probability CSVs if present, otherwise calls `run_leadtime` and caches the result. |
| `get_conversion_month(progression, months)` | Returns the true conversion month — the first visit where `Progression` increases above its baseline value (this is the **corrected** logic; do not use `months[-1]`, which is the last observed visit and can be later than the true conversion visit for subjects with extra follow-up). |
| `analyze_run(prog_prob_df, ctrl_prob_df, threshold, mask_length)` | Scores one model's cached probabilities at a given `threshold`/`mask_length`, applying the mask consistently to both progressors and controls, and returns sensitivity, specificity, mean/median lead time, false-alarm rate, and detection-outcome counts (`n_early`, `n_at_conv`, `n_missed`). |
| `run_grid(prob_dict, thresholds, mask_lengths)` | Runs `analyze_run` for every model × threshold × mask_length combination. |
| `aggregate_grid(df_grid)` | Groups the grid by `(threshold, mask_length)` and computes mean/std/95% CI (via `mean_ci`) across all models for each metric. |

**Note:** `analyze_progressors`/`analyze_controls`/`summarize_leadtime`/`plot_leadtime_summary` are an older, single-model-oriented API (predates the multi-model grid-search functions above) — they now also use the corrected `get_conversion_month` logic, but are largely superseded by `analyze_run`/`run_grid`/`aggregate_grid` for aggregate, cross-model figures.

**How to generate lead-time figures:**
1. Load models with `load_models`, get cached probabilities with `get_probabilities` (or read directly from `leadtime_prob_cache/{imputation}/{cohort}/{model_key}/*.csv` and parse list columns yourself).
2. Call `run_grid` then `aggregate_grid` to get a per-(threshold, mask_length) summary DataFrame across all 10 models/variants.
3. Plot sensitivity, specificity, mean lead time, and false-alarm rate vs. threshold (one line per `mask_length`), or vs. `mask_length` at a fixed threshold. Use distinct line styles/markers per `mask_length` in addition to color, since curves can legitimately overlap (masking only ever removes late, near-conversion truncations, so it can leave early-detection metrics unchanged while still affecting sensitivity/specificity).
4. Aggregated results (already computed) are cached in `lead_time results/leadtime_agg_{imputation}_{cohort}.csv`, with corresponding chart PNGs `lead_time results/leadtime_plots_{imputation}_{cohort}.png`.

---

You now have all the information needed to locate any file, load the data, and produce the requested aggregate figures and statistical summaries.