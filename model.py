import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
# Removed pauc partial AUC dependency; using manual bootstrap for full ROC AUC CI
import os
import itertools
import warnings
import joblib
from joblib import Parallel, delayed
from imblearn.over_sampling import SMOTENC
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
from preprocessing import fit_mmse_imputer, transform_mmse

# Covariates used for MMSE iterative imputation (must be present in the raw DataFrame)
MMSE_COVARIATES = ['EDUC', 'GDS', 'CDR', 'TOBAC30', 'BILLS', 'TAXES', 'SHOPPING',
                   'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES',
                   'TRAVEL', 'hearing', 'vision']

# Categorical feature names used for SMOTENC and encoding
CATEGORICAL_COLS = ['SEX', 'NACCFAM', 'CVHATT', 'CVAFIB', 'DIABETES',
                    'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 'ANX', 'NACCTBI', 'RACE']

def _get_cat_indices(feature_names):
    """Return column indices of categorical features within the feature array."""
    return [i for i, f in enumerate(feature_names) if f in CATEGORICAL_COLS]

def _apply_smotenc(X_train, y_train, cat_indices):
    """Apply SMOTENC to oversample the minority class in the training set.
    Returns (X_resampled, y_resampled). Falls back to original data if SMOTE fails."""
    try:
        min_class_count = int(np.bincount(y_train.astype(int)).min())
        k = min(3, min_class_count - 1) if min_class_count > 1 else 1
        sm = SMOTENC(categorical_features=cat_indices, k_neighbors=k, random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res
    except Exception as e:
        print(f"  SMOTENC failed ({e}); using original training data.")
        return X_train, y_train

# This is the classifer used for visits 2-6 classification scores and charts. 

def create_target_variable_ad(row):
    """Create binary target from Progression column (1=progressed, 0=stable)"""
    progression = eval(row['Progression'])  # Convert string tuple to actual tuple
    return 1 if 2 in progression else 0  # 1 if progressed to AD at any visit
def create_target_variable_cn(row):
    """Create binary target for CN-starting patients.
    1 = progressed to MCI (1) or AD (2) at any visit, 0 = stable CN."""
    progression = eval(row['Progression'])  # Convert string tuple to actual tuple
    return 1 if any(v > 0 for v in progression) else 0

# Keep old name as alias for backwards compatibility
create_target_variable_mci = create_target_variable_cn
def _parse_array_columns(df):
    """Convert array-string columns (e.g. '[1.0, 2.5, nan]') to lists of floats in-place."""
    for col in df.columns:
        if df[col].dtype == object and df[col].str.startswith('[').any():
            df[col] = df[col].apply(
                lambda x: [float(v.strip()) if v.strip() != 'nan' else np.nan
                           for v in x.replace('[', '').replace(']', '').split(',')]
                if isinstance(x, str) and x.startswith('[') else np.nan
            )


def _is_numeric_list_col(series):
    """True if the first element is a list of numeric values."""
    first = series.iloc[0]
    return (isinstance(first, list)
            and all(isinstance(v, (int, float)) for v in first if not pd.isna(v)))


def _has_valid(x):
    return isinstance(x, list) and np.any(~np.isnan(x))


def _first_valid(x):
    if not isinstance(x, list):
        return np.nan
    return next((v for v in x if not np.isnan(v)), np.nan)


def _last_valid(x):
    if not isinstance(x, list):
        return np.nan
    return next((v for v in reversed(x) if not np.isnan(v)), np.nan)


def _calc_slope(x):
    if isinstance(x, list) and len(x) > 1:
        xv = np.arange(len(x))
        yv = np.array(x)
        valid = ~np.isnan(yv)
        if valid.sum() > 1:
            return linregress(xv[valid], yv[valid]).slope
    return np.nan


def _calc_acceleration(x):
    """Slope of consecutive deltas (2nd derivative). Needs >= 3 non-NaN values."""
    if not isinstance(x, list) or len(x) < 3:
        return np.nan
    arr = np.array(x, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 3:
        return np.nan
    # Keep only valid values in order, compute consecutive deltas
    vals = arr[valid]
    deltas = np.diff(vals)
    if len(deltas) < 2:
        return np.nan
    idx = np.arange(len(deltas))
    return linregress(idx, deltas).slope


def _engineer_visit_agnostic(df, new_columns):
    """Populate *new_columns* dict with visit-agnostic features for all numeric-list columns in *df*."""
    for col in df.columns:
        if not _is_numeric_list_col(df[col]):
            continue

        new_columns[f"{col}_mean"] = df[col].apply(
            lambda x: np.nanmean(x) if _has_valid(x) else np.nan)
        new_columns[f"{col}_max"] = df[col].apply(
            lambda x: np.nanmax(x) if _has_valid(x) else np.nan)
        new_columns[f"{col}_min"] = df[col].apply(
            lambda x: np.nanmin(x) if _has_valid(x) else np.nan)
        new_columns[f"{col}_std"] = df[col].apply(
            lambda x: (np.nanstd(x, ddof=1)
                       if isinstance(x, list) and np.sum(~np.isnan(x)) > 1
                       else np.nan))
        new_columns[f"{col}_range"] = df[col].apply(
            lambda x: (np.nanmax(x) - np.nanmin(x)) if _has_valid(x) else np.nan)
        new_columns[f"{col}_slope"] = df[col].apply(_calc_slope)
        new_columns[f"{col}_first"] = df[col].apply(_first_valid)
        new_columns[f"{col}_last"] = df[col].apply(_last_valid)
        new_columns[f"{col}_last_minus_first"] = df[col].apply(
            lambda x: (_last_valid(x) - _first_valid(x))
            if _has_valid(x) else np.nan)
        new_columns[f"{col}_acceleration"] = df[col].apply(_calc_acceleration)
        new_columns[f"{col}_n_visits"] = df[col].apply(
            lambda x: int(np.sum(~np.isnan(x))) if isinstance(x, list) else np.nan)


def create_delta_features(df):
    """Visit-agnostic feature engineering.

    For each longitudinal (list-valued) column, creates fixed-width summary
    statistics that are independent of the number of visits:

        _mean, _max, _min, _std, _range, _slope, _first, _last,
        _last_minus_first, _acceleration, _n_visits

    Static / scalar columns are passed through unchanged.
    """
    df = df.copy()
    new_columns = {}

    _parse_array_columns(df)
    _engineer_visit_agnostic(df, new_columns)

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Drop original array columns (keep only engineered features)
    array_cols = [col for col in df.columns if isinstance(df[col].iloc[0], list)] if len(df) > 0 else []
    df = df.drop(columns=array_cols)
    
    return df

def safe_auc_ci(y_true, y_score, *, n_boot=1000, conf_level=0.95, min_valid=30, random_state=42, verbose=True):
    """Bootstrap 95% CI for full ROC AUC.

    Skips resamples missing a class. Prints how many were skipped if verbose.
    Returns (base_auc, (lower_ci, upper_ci)); CI is (nan, nan) if insufficient valid resamples.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        if verbose:
            print("Bootstrap AUC CI: original sample lacks both classes; skipping.")
        return np.nan, (np.nan, np.nan)
    base_auc = roc_auc_score(y_true, y_score)
    kept = []
    skipped = 0
    n = len(y_true)
    alpha = 1 - conf_level
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        bt_y = y_true[idx]
        if len(np.unique(bt_y)) < 2:
            skipped += 1
            continue
        bt_s = y_score[idx]
        try:
            kept.append(roc_auc_score(bt_y, bt_s))
        except ValueError:
            skipped += 1
            continue
    if verbose:
        print(f"Bootstrap samples: attempted={n_boot}, valid={len(kept)}, skipped={skipped}")
    if len(kept) < min_valid:
        if verbose:
            print(f"Insufficient valid bootstrap samples (<{min_valid}) for CI; returning point estimate only.")
        return base_auc, (np.nan, np.nan)
    kept_arr = np.array(kept)
    lower = np.percentile(kept_arr, 100 * (alpha / 2))
    upper = np.percentile(kept_arr, 100 * (1 - alpha / 2))
    return base_auc, (lower, upper)

def bootstrap_all_metrics_ci(
    y_true,
    y_pred,
    y_score,
    *,
    n_boot=1000,
    conf_level=0.95,
    random_state=42,
    max_attempts=10000,
    verbose=True,
):
    """Bootstrap CIs for accuracy, precision (macro), recall (macro), F1 (macro), and ROC AUC.

    Keeps sampling until n_boot valid (both-class) resamples are collected, up to max_attempts.
    Returns a dict including a 'valid_samples' key showing how many attempts were needed:
      {
        'accuracy': (point, (lo, hi)),
        'precision_macro': (point, (lo, hi)),
        'recall_macro': (point, (lo, hi)),
        'f1_macro': (point, (lo, hi)),
        'auc': (point, (lo, hi)),
        'valid_samples': 'n_boot/total_attempts'  # e.g. '1000/1050'
      }
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)
    alpha = 1 - conf_level

    # Point estimates
    acc_pt = accuracy_score(y_true, y_pred)
    prec_pt = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_pt = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_pt = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc_pt = np.nan
    if len(np.unique(y_true)) == 2:
        auc_pt = roc_auc_score(y_true, y_score)

    # Bootstrap arrays
    acc_b, prec_b, rec_b, f1_b = [], [], [], []
    auc_b = []
    n = len(y_true)
    total_attempts = 0
    skipped = 0

    # Keep sampling until we have n_boot valid samples (or hit max_attempts)
    while len(auc_b) < n_boot and total_attempts < max_attempts:
        total_attempts += 1
        idx = rng.choice(n, n, replace=True)
        bt_y = y_true[idx]

        # Skip if resample lacks both classes
        if len(np.unique(bt_y)) < 2:
            skipped += 1
            continue

        bt_pred = y_pred[idx]
        bt_score = y_score[idx]

        acc_b.append(accuracy_score(bt_y, bt_pred))
        prec_b.append(precision_score(bt_y, bt_pred, average="macro", zero_division=0))
        rec_b.append(recall_score(bt_y, bt_pred, average="macro", zero_division=0))
        f1_b.append(f1_score(bt_y, bt_pred, average="macro", zero_division=0))

        try:
            auc_b.append(roc_auc_score(bt_y, bt_score))
        except ValueError:
            # Shouldn't happen if unique check passed, but guard anyway
            skipped += 1
            # Remove the metrics we just appended since AUC failed
            acc_b.pop()
            prec_b.pop()
            rec_b.pop()
            f1_b.pop()

    valid_samples_str = f"{len(auc_b)}/{total_attempts}"

    if verbose:
        print(f"Bootstrap: valid={len(auc_b)}, total_attempts={total_attempts}, skipped={skipped} ({valid_samples_str})")

    # Helper to CI from list
    def ci_bounds(vals):
        arr = np.array(vals)
        return (
            float(np.percentile(arr, 100 * (alpha / 2))),
            float(np.percentile(arr, 100 * (1 - alpha / 2))),
        )

    results = {
        "accuracy": (acc_pt, ci_bounds(acc_b) if acc_b else (np.nan, np.nan)),
        "precision_macro": (prec_pt, ci_bounds(prec_b) if prec_b else (np.nan, np.nan)),
        "recall_macro": (rec_pt, ci_bounds(rec_b) if rec_b else (np.nan, np.nan)),
        "f1_macro": (f1_pt, ci_bounds(f1_b) if f1_b else (np.nan, np.nan)),
        "auc": (auc_pt, ci_bounds(auc_b) if auc_b else (np.nan, np.nan)),
        "valid_samples": valid_samples_str,
    }

    return results

# Preprocess the data
def preprocess_data(df, progression_type):
    # Create target variable if not already present
    if 'target' not in df.columns:
        if progression_type == 'AD':
            df['target'] = df.apply(create_target_variable_ad, axis=1)
        elif progression_type in ('MCI', 'CN'):
            df['target'] = df.apply(create_target_variable_cn, axis=1)
    
    # Get all available features (after create_delta_features transformation)
    all_features = [col for col in df.columns if col != 'target']
    
    # Select features we want to keep
    # Static features (non-time-series)
    static_features = [
        'SEX', 'EDUC', 'ALCOHOL', 'NACCFAM', 'CVHATT', 
        'CVAFIB', 'DIABETES', 'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 
        'ANX', 'NACCTBI', 'SMOKYRS', 'RACE', 'age', 'HISPANIC'
    ]
    
    # Time-series features — visit-agnostic summary statistics
    _TS_SUFFIXES = ('_slope', '_mean', '_max', '_min', '_std', '_range',
                    '_first', '_last', '_last_minus_first', '_acceleration',
                    '_n_visits')
    time_series_features = [col for col in all_features if any(s in col for s in _TS_SUFFIXES)]
    
    # Only keep features that actually exist in the dataframe
    static_features = [f for f in static_features if f in df.columns]
    
    # Combine all features
    features = static_features + time_series_features
    
    # Handle missing values
    df = df[features + ['target']].copy()
    df = df.dropna(subset=features, how='all')
    
    # Convert categorical variables
    categorical_cols = ['SEX', 'NACCFAM', 'CVHATT', 'CVAFIB', 'DIABETES', 
                       'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 'ANX', 'NACCTBI', 'RACE']
    categorical_cols = [col for col in categorical_cols if col in df.columns]  # Only keep existing cols
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes  # Convert to numeric codes
    
    # Return processed DataFrame without imputation/scaling to avoid leakage
    # Keep return signature compatibility (scaler, imputer as None)
    return df, None, None


def aggregate_original_features(df):
    """Aggregate time-series features using mean and combine with static features"""
    df = df.copy()
    
    # Parse array strings into lists
    for col in df.columns:
        if df[col].dtype == object and df[col].str.startswith('[').any():
            df[col] = df[col].apply(
                lambda x: [float(v.strip()) if v.strip() != 'nan' else np.nan 
                         for v in x.replace('[','').replace(']','').split(',')] 
                if isinstance(x, str) and x.startswith('[') else np.nan
            )
    
    # Aggregate time-series features using mean
    aggregated_features = {}
    time_series_cols = [col for col in df.columns if isinstance(df[col].iloc[0], list)]
    
    for col in time_series_cols:
        aggregated_features[col] = df[col].apply(
            lambda x: np.nanmean(x) if isinstance(x, list) else np.nan
        )
    
    # Static features (non-time-series)
    static_features = [
        'SEX', 'EDUC', 'ALCOHOL', 'NACCFAM', 'CVHATT', 
        'CVAFIB', 'DIABETES', 'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 
        'ANX', 'NACCTBI', 'SMOKYRS', 'RACE', 'age'
    ]
    static_features = [f for f in static_features if f in df.columns]
    
    # Combine aggregated features with static features
    aggregated_df = pd.DataFrame(aggregated_features)
    static_df = df[static_features]
    combined_df = pd.concat([static_df, aggregated_df], axis=1)
    
    return combined_df

def split_dataset(data):
    """Split dataset into training and testing sets. For only the training set, recalculate all scalers and imputers."""
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Return raw splits; scaling/imputation will be fit on train only downstream
    return X_train.values, X_test.values, y_train.values, y_test.values

# For the training of the best model with the best set of hyperparameters, prints out the whole performance report.
def build_model_final(X_train, X_test, y_train, y_test, model_dict, feature_names, use_smote=True):
    
    # Fit imputer/scaler on training set only (avoid leakage)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_proc = scaler.transform(imputer.transform(X_test))

    # Optionally apply SMOTENC on processed training data to handle class imbalance
    if use_smote:
        cat_indices = _get_cat_indices(feature_names)
        X_train_res, y_train_res = _apply_smotenc(X_train_proc, y_train, cat_indices)
        print(f"SMOTE resampling: {len(y_train)} -> {len(y_train_res)} training samples")
        spw = 1.0
    else:
        X_train_res, y_train_res = X_train_proc, y_train
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"SMOTE disabled: using {len(y_train)} training samples as-is (scale_pos_weight={spw:.2f})")

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',

        n_estimators=model_dict['n_estimators'],
        max_depth=model_dict['max_depth'],
        learning_rate=model_dict['learning_rate'],
        subsample=model_dict['subsample'],
        colsample_bytree=model_dict['colsample_bytree'],

        scale_pos_weight=spw,
        random_state=42,
    )

    model.fit(X_train_res, y_train_res)
    
    # Evaluate model
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]
    
    print("Classification Report:")
    cr_str = classification_report(y_test, y_pred)
    print(cr_str)
    
    # Base ROC AUC
    base_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC AUC Score: {base_auc:.4f}")

    # Bootstrap CIs for all metrics
    metrics = bootstrap_all_metrics_ci(
        y_test,
        y_pred,
        y_proba,
        n_boot=1000,
        conf_level=0.95,
        random_state=42,
        verbose=True,
    )

    acc_pt, (acc_lo, acc_hi) = metrics["accuracy"]
    prec_pt, (prec_lo, prec_hi) = metrics["precision_macro"]
    rec_pt, (rec_lo, rec_hi) = metrics["recall_macro"]
    f1_pt, (f1_lo, f1_hi) = metrics["f1_macro"]
    auc_pt, (auc_lo, auc_hi) = metrics["auc"]

    print(f"\nBootstrap 95% CI (n=1000, valid_samples={metrics['valid_samples']}):")
    print(f"- Accuracy: {acc_pt:.3f} (CI: {acc_lo:.3f}, {acc_hi:.3f}) range={acc_hi - acc_lo:.3f}")
    print(f"- Precision (macro): {prec_pt:.3f} (CI: {prec_lo:.3f}, {prec_hi:.3f}) range={prec_hi - prec_lo:.3f}")
    print(f"- Recall (macro): {rec_pt:.3f} (CI: {rec_lo:.3f}, {rec_hi:.3f}) range={rec_hi - rec_lo:.3f}")
    print(f"- F1 (macro): {f1_pt:.3f} (CI: {f1_lo:.3f}, {f1_hi:.3f}) range={f1_hi - f1_lo:.3f}")
    if not np.isnan(auc_lo):
        print(f"- ROC AUC: {auc_pt:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f}) range={auc_hi - auc_lo:.3f}")
    else:
        print(f"- ROC AUC: {auc_pt:.3f} (CI unavailable; too few valid resamples)")

    summary = {
        "classification_report": cr_str,
        "base_auc": float(base_auc),
        "bootstrap_metrics": metrics,
    }
    return model, feature_names, imputer, scaler, summary

# To train models for cross-validation. Returns only AUC score. Does not print out anything.
def build_model(X_train, X_test, y_train, y_test, model_dict, feature_names, use_smote=True, xgb_n_jobs=None):
    
    # Per-fold imputation/scaling fit on training fold only (avoid leakage)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_proc = scaler.transform(imputer.transform(X_test))

    # Optionally apply SMOTENC on processed training fold to handle class imbalance
    if use_smote:
        cat_indices = _get_cat_indices(feature_names)
        X_train_res, y_train_res = _apply_smotenc(X_train_proc, y_train, cat_indices)
        spw = 1.0
    else:
        X_train_res, y_train_res = X_train_proc, y_train
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',

        n_estimators=model_dict['n_estimators'],
        max_depth=model_dict['max_depth'],
        learning_rate=model_dict['learning_rate'],
        subsample=model_dict['subsample'],
        colsample_bytree=model_dict['colsample_bytree'],

        scale_pos_weight=spw,
        n_jobs=xgb_n_jobs,
        random_state=42,
    )

    model.fit(X_train_res, y_train_res)
    
    # Evaluate model
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    return roc_auc_score(y_test, y_proba)


def _loocv_fold(fold_idx, df_with_target, covariates, mmse_needs_imputation,
                hyperparams, use_smote, progression_type):
    """Run one LOO fold from raw data through prediction.

    The full pipeline (MMSE imputation -> delta features -> preprocess ->
    impute/scale -> optional SMOTE -> XGBoost) is executed so that the MMSE
    imputer is never fit on the held-out sample.

    Returns (true_label, predicted_proba, predicted_class).
    """
    n_est, max_depth, lr, subsample, colsample = hyperparams

    # 1. train / test split
    all_idx = list(range(len(df_with_target)))
    train_idx = all_idx[:fold_idx] + all_idx[fold_idx + 1:]
    df_train = df_with_target.iloc[train_idx].copy()
    df_test  = df_with_target.iloc[[fold_idx]].copy()
    true_label = int(df_test['target'].iloc[0])

    # 2. MMSE imputation (fit on N-1 only)
    if mmse_needs_imputation:
        mmse_imp, df_train = fit_mmse_imputer(df_train, covariates)
        df_test = transform_mmse(df_test, covariates, mmse_imp)

    # 3. Separate targets, drop from DataFrames
    y_tr = df_train['target'].values
    y_te_val = df_test['target'].values
    df_train = df_train.drop(columns=['target'])
    df_test  = df_test.drop(columns=['target'])

    # 4. Feature engineering
    df_train = create_delta_features(df_train)
    df_test  = create_delta_features(df_test)

    # 5. Re-attach targets for preprocess_data
    df_train['target'] = y_tr
    df_test['target']  = y_te_val
    processed_train, _, _ = preprocess_data(df_train, progression_type)
    processed_test,  _, _ = preprocess_data(df_test, progression_type)

    # 6. Align columns (test may lack some delta columns present in train)
    feat_names = [c for c in processed_train.columns if c != 'target']
    for col in feat_names:
        if col not in processed_test.columns:
            processed_test[col] = np.nan
    processed_test = processed_test[feat_names + ['target']]

    X_tr = processed_train.drop(columns=['target']).values
    X_te = processed_test.drop(columns=['target']).values
    y_tr = processed_train['target'].values

    # 7. Impute + scale
    imp = SimpleImputer(strategy='mean')
    scl = StandardScaler()
    X_tr = scl.fit_transform(imp.fit_transform(X_tr))
    X_te = scl.transform(imp.transform(X_te))

    # 8. Optional SMOTE
    if use_smote:
        cat_idx = _get_cat_indices(feat_names)
        X_tr, y_tr = _apply_smotenc(X_tr, y_tr, cat_idx)

    # 9. Train + predict
    # When SMOTE is disabled (Mode A), compute scale_pos_weight from the N-1
    # training labels to up-weight the minority class inside XGBoost's loss.
    # Computed per fold from y_tr only — no leakage from the held-out sample.
    if not use_smote:
        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
    else:
        spw = 1.0

    clf = XGBClassifier(
        objective='binary:logistic', eval_metric='auc',
        n_estimators=n_est, max_depth=max_depth, learning_rate=lr,
        subsample=subsample, colsample_bytree=colsample,
        scale_pos_weight=spw,
        n_jobs=1, random_state=42,
    )
    clf.fit(X_tr, y_tr)
    proba = float(clf.predict_proba(X_te)[:, 1][0])
    pred  = int(clf.predict(X_te)[0])

    return (true_label, proba, pred, feat_names, clf.feature_importances_)


def _eval_loocv_combo_pipeline(combo, df_with_target, covariates,
                                mmse_needs_imputation, use_smote, progression_type):
    """Evaluate one hyperparameter combo across all LOO folds (full pipeline per fold).
    Returns result tuple: (n_est, max_depth, lr, subsample, colsample, auc_score).
    """
    warnings.filterwarnings('ignore')
    n_samples = len(df_with_target)
    results = []
    for i in range(n_samples):
        results.append(_loocv_fold(
            i, df_with_target, covariates, mmse_needs_imputation,
            combo, use_smote, progression_type,
        ))
    y_true  = np.array([r[0] for r in results])
    y_score = np.array([r[1] for r in results])
    try:
        score = roc_auc_score(y_true, y_score)
    except Exception:
        score = 0.0
    return (*combo, score)


def _loocv_final_evaluation(df_with_target, best_params, covariates,
                             mmse_needs_imputation, use_smote, progression_type,
                             n_jobs=1):
    """Re-run LOOCV with best hyperparameters; print full report with bootstrap CIs.
    Returns a summary dict compatible with the report writer in train_best_model.

    n_jobs : int
        Number of parallel workers for fold evaluation (1 = serial).
    """
    combo = (best_params['n_estimators'], best_params['max_depth'],
             best_params['learning_rate'], best_params['subsample'],
             best_params['colsample_bytree'])
    n_samples = len(df_with_target)

    if n_jobs == 1:
        results = []
        for i in tqdm(range(n_samples), desc="LOOCV final evaluation", unit="fold"):
            results.append(_loocv_fold(
                i, df_with_target, covariates, mmse_needs_imputation,
                combo, use_smote, progression_type,
            ))
    else:
        print(f"Running LOOCV final evaluation in parallel (n_jobs={n_jobs}, {n_samples} folds)...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_loocv_fold)(
                i, df_with_target, covariates, mmse_needs_imputation,
                combo, use_smote, progression_type,
            )
            for i in tqdm(range(n_samples), desc="LOOCV final evaluation", unit="fold")
        )

    y_true  = np.array([r[0] for r in results])
    y_proba = np.array([r[1] for r in results])
    y_pred  = np.array([r[2] for r in results])

    # Classification report
    cr_str = classification_report(y_true, y_pred)
    print("Classification Report (LOOCV):")
    print(cr_str)

    # ROC AUC
    base_auc = roc_auc_score(y_true, y_proba)
    print(f"\nROC AUC Score: {base_auc:.4f}")

    # Bootstrap CIs
    metrics = bootstrap_all_metrics_ci(
        y_true, y_pred, y_proba,
        n_boot=1000, conf_level=0.95, random_state=42, verbose=True,
    )
    acc_pt,  (acc_lo,  acc_hi)  = metrics["accuracy"]
    prec_pt, (prec_lo, prec_hi) = metrics["precision_macro"]
    rec_pt,  (rec_lo,  rec_hi)  = metrics["recall_macro"]
    f1_pt,   (f1_lo,   f1_hi)   = metrics["f1_macro"]
    auc_pt,  (auc_lo,  auc_hi)  = metrics["auc"]

    print(f"\nBootstrap 95% CI (n=1000, valid_samples={metrics['valid_samples']}):")
    print(f"- Accuracy: {acc_pt:.3f} (CI: {acc_lo:.3f}, {acc_hi:.3f}) range={acc_hi - acc_lo:.3f}")
    print(f"- Precision (macro): {prec_pt:.3f} (CI: {prec_lo:.3f}, {prec_hi:.3f}) range={prec_hi - prec_lo:.3f}")
    print(f"- Recall (macro): {rec_pt:.3f} (CI: {rec_lo:.3f}, {rec_hi:.3f}) range={rec_hi - rec_lo:.3f}")
    print(f"- F1 (macro): {f1_pt:.3f} (CI: {f1_lo:.3f}, {f1_hi:.3f}) range={f1_hi - f1_lo:.3f}")
    if not np.isnan(auc_lo):
        print(f"- ROC AUC: {auc_pt:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f}) range={auc_hi - auc_lo:.3f}")
    else:
        print(f"- ROC AUC: {auc_pt:.3f} (CI unavailable; too few valid resamples)")

    # Aggregate feature importances across folds
    feat_names = results[0][3]
    all_importances = np.array([r[4] for r in results])
    mean_importances = all_importances.mean(axis=0)

    return {
        "classification_report": cr_str,
        "base_auc": float(base_auc),
        "bootstrap_metrics": metrics,
        "best_params": best_params,
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "feature_names": list(feat_names),
        "feature_importances": mean_importances,
    }


def _eval_skf_combo(combo, list_x_train, list_x_test, list_y_train, list_y_test, n_splits, feature_names, use_smote):
    """Evaluate one hyperparameter combo across K folds. Returns result tuple."""
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree = combo
    model_dict = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
    }
    score_sum = 0
    for i in range(n_splits):
        score_sum += build_model(
            list_x_train[i], list_x_test[i],
            list_y_train[i], list_y_test[i],
            model_dict, feature_names,
            use_smote=use_smote, xgb_n_jobs=1,
        )
    score = score_sum / n_splits
    return (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, score)


# Perform grid search with the training dataset and the given parameter ranges.
# Return the set of best hyperparameters and save cross-validation scores to the given csv path.
# A helper method for train_best_model(...)
#
# cv_method : 'skf'   → StratifiedKFold (default)
#             'loocv' → Leave-One-Out CV
# use_smote : whether to oversample the minority class inside each CV fold/iteration
# n_jobs    : number of parallel workers for evaluating hyperparameter combos (1 = serial)
def grid_search(x, y, param_grid, csv_path, feature_names, cv_method='skf', use_smote=True, n_jobs=1,
                df_raw=None, covariates=None, mmse_needs_imputation=False, progression_type=None):
    best_hyperparameters = None
    best_score = 0

    combo_keys = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
    all_combos = list(itertools.product(*(param_grid[k] for k in combo_keys)))
    total_combos = len(all_combos)

    # ── LOOCV branch (full pipeline per fold) ────────────────────────────────────
    if cv_method == 'loocv':
        n_samples = len(df_raw)
        print(f"Using Leave-One-Out CV ({n_samples} iterations x {total_combos} combos, use_smote={use_smote}, n_jobs={n_jobs})")

        results = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_eval_loocv_combo_pipeline)(
                combo, df_raw, covariates, mmse_needs_imputation, use_smote, progression_type,
            )
            for combo in all_combos
        )
        scores = []
        with tqdm(total=total_combos, desc="LOOCV grid search", unit="combo") as pbar:
            for result in results:
                scores.append(list(result))
                if result[5] > best_score:
                    best_score = result[5]
                    best_hyperparameters = dict(zip(combo_keys, result[:5]))
                pbar.update(1)

    # ── StratifiedKFold branch ───────────────────────────────────────────────────
    else:
        y_arr = np.array(y)
        list_x_train = []
        list_x_test = []
        list_y_train = []
        list_y_test = []
        # Determine feasible n_splits based on minority class count
        if np.unique(y_arr).size == 2:
            class_counts = np.bincount(y_arr)
            min_class = int(class_counts.min())
        else:
            min_class = len(y_arr)
        n_splits = max(2, min(5, min_class))
        if n_splits < 5:
            print(f"Note: Using StratifiedKFold with n_splits={n_splits} due to minority class size={min_class} (use_smote={use_smote})")
        else:
            print(f"Using StratifiedKFold with n_splits={n_splits} (use_smote={use_smote})")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for i, (train_ind, test_ind) in enumerate(skf.split(x, y_arr)):
            list_x_train.append(x[train_ind])
            list_y_train.append(y_arr[train_ind])
            list_x_test.append(x[test_ind])
            list_y_test.append(y_arr[test_ind])

        print(f"Grid search: {total_combos} hyperparameter combinations (n_jobs={n_jobs})")

        results = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_eval_skf_combo)(combo, list_x_train, list_x_test, list_y_train, list_y_test, n_splits, feature_names, use_smote)
            for combo in all_combos
        )
        scores = []
        with tqdm(total=total_combos, desc="SKF grid search", unit="combo") as pbar:
            for result in results:
                scores.append(list(result))
                if result[5] > best_score:
                    best_score = result[5]
                    best_hyperparameters = dict(zip(combo_keys, result[:5]))
                pbar.update(1)

    # ensure directory exists and save the cross validations scores as csv
    dirn = os.path.dirname(csv_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    pd.DataFrame(scores, columns=["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "avg AUC score"]).to_csv(csv_path, index=False)
    return best_hyperparameters

# Input an unprocessed dataset, progression type for preprocessing, and the parameter ranges 
# to get a best performing model with the best set of hyperparameters found from grid-search.
# Save cross-validation scores to the input new csv path.
#
# cv_method  : 'skf'   → StratifiedKFold grid search (default)
#              'loocv' → Leave-One-Out CV grid search
# use_smote  : whether to apply SMOTENC inside each CV fold (grid search) and in the final model
#
# LOOCV modes:
#   Mode A (loocv + no SMOTE) : no train/test split; LOOCV on full dataset IS the evaluation.
#   Mode B (loocv + SMOTE)    : 80/20 split; LOOCV grid search on training set (per-fold MMSE);
#                                final model evaluated on holdout via build_model_final.
def train_best_model(dataset, progression_type, param_grid, csv_path, save_dir="saved_models", model_base_name=None, save_artifacts=True, cv_method='skf', use_smote=True, n_jobs=1):
    
    dataset = dataset.copy()

    # --- Step 1: Create target variable on the raw DataFrame (needed for stratified split) ---
    if progression_type == 'AD':
        dataset['target'] = dataset.apply(create_target_variable_ad, axis=1)
    elif progression_type in ('MCI', 'CN'):
        dataset['target'] = dataset.apply(create_target_variable_cn, axis=1)

    # Determine MMSE imputation needs early (shared by all modes)
    covariates = [c for c in MMSE_COVARIATES if c in dataset.columns]
    mmse_needs_imputation = (
        'MMSE' in dataset.columns
        and dataset['MMSE'].astype(str).str.contains('nan', na=False).any()
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  Mode A: LOOCV without SMOTE  (no train/test split)
    # ══════════════════════════════════════════════════════════════════════════
    if cv_method == 'loocv' and not use_smote:
        print(f"\n{'='*60}")
        print(f"Mode A: LOOCV without SMOTE — full dataset ({len(dataset)} samples)")
        print(f"MMSE imputation: {'per-fold (fit on N-1)' if mmse_needs_imputation else 'not needed'}")
        print(f"{'='*60}")

        # Grid search with per-fold pipeline (MMSE imputed inside each LOO fold)
        model_dict = grid_search(
            None, None, param_grid, csv_path, None,
            cv_method='loocv', use_smote=False, n_jobs=n_jobs,
            df_raw=dataset, covariates=covariates,
            mmse_needs_imputation=mmse_needs_imputation,
            progression_type=progression_type,
        )

        # Final evaluation: re-run LOOCV with best hyperparameters
        summary = _loocv_final_evaluation(
            dataset, model_dict, covariates, mmse_needs_imputation,
            use_smote=False, progression_type=progression_type,
            n_jobs=n_jobs,
        )

        # Save report only (LOOCV produces N models; no single model to persist)
        if save_artifacts:
            os.makedirs(save_dir, exist_ok=True)
            base = model_base_name or os.path.splitext(os.path.basename(csv_path))[0]
            report_path = os.path.join(save_dir, f"{base}_report_{progression_type}.txt")
            with open(report_path, "w") as f:
                f.write(f"Dataset base: {base}\n")
                f.write(f"Progression type: {progression_type}\n")
                f.write(f"CV method: loocv (Mode A — no train/test split)\n")
                f.write(f"Class balancing: disabled\n")
                f.write(f"MMSE imputation: {'per-fold (fit on N-1)' if mmse_needs_imputation else 'not needed'}\n")
                f.write(f"Total samples: {len(dataset)}\n")
                f.write("\nBest hyperparameters:\n")
                for k, v in summary['best_params'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\nClassification Report:\n")
                f.write(summary["classification_report"] + "\n")
                f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
                bm = summary["bootstrap_metrics"]
                f.write(f"Bootstrap 95% CI (n=1000, valid_samples={bm['valid_samples']}):\n")
                f.write(f"- Accuracy: {bm['accuracy'][0]:.3f} (CI: {bm['accuracy'][1][0]:.3f}, {bm['accuracy'][1][1]:.3f}) range={bm['accuracy'][1][1] - bm['accuracy'][1][0]:.3f}\n")
                f.write(f"- Precision (macro): {bm['precision_macro'][0]:.3f} (CI: {bm['precision_macro'][1][0]:.3f}, {bm['precision_macro'][1][1]:.3f}) range={bm['precision_macro'][1][1] - bm['precision_macro'][1][0]:.3f}\n")
                f.write(f"- Recall (macro): {bm['recall_macro'][0]:.3f} (CI: {bm['recall_macro'][1][0]:.3f}, {bm['recall_macro'][1][1]:.3f}) range={bm['recall_macro'][1][1] - bm['recall_macro'][1][0]:.3f}\n")
                f.write(f"- F1 (macro): {bm['f1_macro'][0]:.3f} (CI: {bm['f1_macro'][1][0]:.3f}, {bm['f1_macro'][1][1]:.3f}) range={bm['f1_macro'][1][1] - bm['f1_macro'][1][0]:.3f}\n")
                auc_lo, auc_hi = bm['auc'][1]
                if not np.isnan(auc_lo):
                    f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f}) range={auc_hi - auc_lo:.3f}\n")
                else:
                    f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI unavailable; too few valid resamples)\n")
            print(f"\nSaved report: {report_path}")

        return None, None  # No single model in Mode A

    # ══════════════════════════════════════════════════════════════════════════
    #  Mode B: LOOCV with SMOTE  (80/20 split)
    # ══════════════════════════════════════════════════════════════════════════
    elif cv_method == 'loocv' and use_smote:
        print(f"\n{'='*60}")
        print(f"Mode B: LOOCV with SMOTE — 80/20 split ({len(dataset)} total)")
        print(f"MMSE imputation: {'per-fold in grid search; full-train for final model' if mmse_needs_imputation else 'not needed'}")
        print(f"{'='*60}")

        # --- 80/20 stratified split on raw DataFrame ---
        train_idx, test_idx = train_test_split(
            dataset.index, test_size=0.2, random_state=42, stratify=dataset['target'],
        )
        df_train_raw = dataset.loc[train_idx].copy()
        df_test_raw  = dataset.loc[test_idx].copy()

        # --- MMSE imputation for FINAL MODEL (fit on full training set) ---
        mmse_imputer_obj = None
        if mmse_needs_imputation:
            print(f"Fitting MMSE imputer on training set ({len(df_train_raw)} samples) with {len(covariates)} covariates...")
            mmse_imputer_obj, df_train_imputed = fit_mmse_imputer(df_train_raw, covariates)
            df_test_imputed = transform_mmse(df_test_raw, covariates, mmse_imputer_obj)
            print("MMSE imputation complete (train fit, test transformed).")
        else:
            df_train_imputed = df_train_raw
            df_test_imputed  = df_test_raw

        # --- Process imputed data for the FINAL MODEL ---
        y_train_target = df_train_imputed['target'].values
        y_test_target  = df_test_imputed['target'].values
        df_tr_feat = create_delta_features(df_train_imputed.drop(columns=['target']))
        df_te_feat = create_delta_features(df_test_imputed.drop(columns=['target']))
        df_tr_feat['target'] = y_train_target
        df_te_feat['target'] = y_test_target
        processed_train, _, _ = preprocess_data(df_tr_feat, progression_type)
        processed_test,  _, _ = preprocess_data(df_te_feat, progression_type)

        feature_names = [c for c in processed_train.columns if c != 'target']
        for col in feature_names:
            if col not in processed_test.columns:
                processed_test[col] = np.nan
        processed_test = processed_test[feature_names + ['target']]

        X_train = processed_train.drop(columns=['target']).values
        X_test  = processed_test.drop(columns=['target']).values
        y_train = processed_train['target'].values
        y_test  = processed_test['target'].values

        # --- Grid search: LOOCV on RAW training data (per-fold MMSE inside each LOO fold) ---
        model_dict = grid_search(
            None, None, param_grid, csv_path, None,
            cv_method='loocv', use_smote=True, n_jobs=n_jobs,
            df_raw=df_train_raw, covariates=covariates,
            mmse_needs_imputation=mmse_needs_imputation,
            progression_type=progression_type,
        )

        # --- Final model trained on full processed training set, evaluated on holdout ---
        model, columns, imputer, scaler, summary = build_model_final(
            X_train, X_test, y_train, y_test, model_dict, feature_names, use_smote=True,
        )

        try:
            model.feature_names_in_ = np.array(columns)
        except Exception:
            pass

        if save_artifacts:
            os.makedirs(save_dir, exist_ok=True)
            base = model_base_name or os.path.splitext(os.path.basename(csv_path))[0]
            model_path   = os.path.join(save_dir, f"{base}_model_{progression_type}.pkl")
            scaler_path  = os.path.join(save_dir, f"{base}_scaler_{progression_type}.pkl")
            imputer_path = os.path.join(save_dir, f"{base}_imputer_{progression_type}.pkl")

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(imputer, imputer_path)

            print("\nSaved artifacts:")
            print(f"- Model:   {model_path}")
            print(f"- Scaler:  {scaler_path}")
            print(f"- Imputer: {imputer_path}")

            if mmse_imputer_obj is not None:
                mmse_imp_path = os.path.join(save_dir, f"{base}_mmse_imputer_{progression_type}.pkl")
                joblib.dump(mmse_imputer_obj, mmse_imp_path)
                print(f"- MMSE Imputer: {mmse_imp_path}")

            report_path = os.path.join(save_dir, f"{base}_report_{progression_type}.txt")
            with open(report_path, "w") as f:
                f.write(f"Dataset base: {base}\n")
                f.write(f"Progression type: {progression_type}\n")
                f.write(f"CV method: loocv (Mode B — 80/20 split + SMOTE)\n")
                f.write(f"MMSE imputation: {'per-fold in grid search; full-train for final model' if mmse_imputer_obj else 'not needed'}\n")
                f.write(f"Class balancing: SMOTENC\n")
                f.write("Best hyperparameters:\n")
                for k, v in model.get_params().items():
                    if k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state", "objective", "eval_metric"]:
                        f.write(f"  {k}: {v}\n")
                f.write("\nClassification Report:\n")
                f.write(summary["classification_report"] + "\n")
                f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
                bm = summary["bootstrap_metrics"]
                f.write(f"Bootstrap 95% CI (n=1000, valid_samples={bm['valid_samples']}):\n")
                f.write(f"- Accuracy: {bm['accuracy'][0]:.3f} (CI: {bm['accuracy'][1][0]:.3f}, {bm['accuracy'][1][1]:.3f}) range={bm['accuracy'][1][1] - bm['accuracy'][1][0]:.3f}\n")
                f.write(f"- Precision (macro): {bm['precision_macro'][0]:.3f} (CI: {bm['precision_macro'][1][0]:.3f}, {bm['precision_macro'][1][1]:.3f}) range={bm['precision_macro'][1][1] - bm['precision_macro'][1][0]:.3f}\n")
                f.write(f"- Recall (macro): {bm['recall_macro'][0]:.3f} (CI: {bm['recall_macro'][1][0]:.3f}, {bm['recall_macro'][1][1]:.3f}) range={bm['recall_macro'][1][1] - bm['recall_macro'][1][0]:.3f}\n")
                f.write(f"- F1 (macro): {bm['f1_macro'][0]:.3f} (CI: {bm['f1_macro'][1][0]:.3f}, {bm['f1_macro'][1][1]:.3f}) range={bm['f1_macro'][1][1] - bm['f1_macro'][1][0]:.3f}\n")
                auc_lo, auc_hi = bm['auc'][1]
                if not np.isnan(auc_lo):
                    f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f}) range={auc_hi - auc_lo:.3f}\n")
                else:
                    f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI unavailable; too few valid resamples)\n")
            print(f"- Report:  {report_path}")

        return model, columns

    # ══════════════════════════════════════════════════════════════════════════
    #  SKF (unchanged default behaviour)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        # --- Step 2: Stratified train/test split on raw DataFrame indices ---
        train_idx, test_idx = train_test_split(
            dataset.index, test_size=0.2, random_state=42, stratify=dataset['target']
        )
        df_train_raw = dataset.loc[train_idx].copy()
        df_test_raw = dataset.loc[test_idx].copy()

        # --- Step 3: MMSE imputation post-split (fit on train only) ---
        mmse_imputer_obj = None
        if mmse_needs_imputation:
            print(f"Fitting MMSE imputer on training set ({len(df_train_raw)} samples) with {len(covariates)} covariates...")
            mmse_imputer_obj, df_train_raw = fit_mmse_imputer(df_train_raw, covariates)
            df_test_raw = transform_mmse(df_test_raw, covariates, mmse_imputer_obj)
            print("MMSE imputation complete (train fit, test transformed).")
        else:
            print("No MMSE NaN values found; skipping MMSE imputation.")

        # --- Step 4: Feature engineering + preprocessing on each split independently ---
        y_train_target = df_train_raw['target'].values
        y_test_target = df_test_raw['target'].values
        df_train_raw = df_train_raw.drop(columns=['target'])
        df_test_raw = df_test_raw.drop(columns=['target'])

        df_train_feat = create_delta_features(df_train_raw)
        df_test_feat = create_delta_features(df_test_raw)

        df_train_feat['target'] = y_train_target
        df_test_feat['target'] = y_test_target

        processed_train, _, _ = preprocess_data(df_train_feat, progression_type)
        processed_test, _, _ = preprocess_data(df_test_feat, progression_type)

        feature_names = [c for c in processed_train.columns if c != 'target']
        for col in feature_names:
            if col not in processed_test.columns:
                processed_test[col] = np.nan
        processed_test = processed_test[feature_names + ['target']]

        X_train = processed_train.drop(columns=['target']).values
        X_test = processed_test.drop(columns=['target']).values
        y_train = processed_train['target'].values
        y_test = processed_test['target'].values

        # --- Step 5: Grid search (uses SMOTE inside each CV fold) ---
        model_dict = grid_search(X_train, y_train, param_grid, csv_path, feature_names, cv_method=cv_method, use_smote=use_smote, n_jobs=n_jobs)

        # --- Step 6: Final model with full report ---
        model, columns, imputer, scaler, summary = build_model_final(X_train, X_test, y_train, y_test, model_dict, feature_names, use_smote=use_smote)

        try:
            model.feature_names_in_ = np.array(columns)
        except Exception:
            pass

        if save_artifacts:
            os.makedirs(save_dir, exist_ok=True)
            base = model_base_name or os.path.splitext(os.path.basename(csv_path))[0]
            model_path = os.path.join(save_dir, f"{base}_model_{progression_type}.pkl")
            scaler_path = os.path.join(save_dir, f"{base}_scaler_{progression_type}.pkl")
            imputer_path = os.path.join(save_dir, f"{base}_imputer_{progression_type}.pkl")

            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(imputer, imputer_path)

            print("\nSaved artifacts:")
            print(f"- Model:   {model_path}")
            print(f"- Scaler:  {scaler_path}")
            print(f"- Imputer: {imputer_path}")

            if mmse_imputer_obj is not None:
                mmse_imp_path = os.path.join(save_dir, f"{base}_mmse_imputer_{progression_type}.pkl")
                joblib.dump(mmse_imputer_obj, mmse_imp_path)
                print(f"- MMSE Imputer: {mmse_imp_path}")

            report_path = os.path.join(save_dir, f"{base}_report_{progression_type}.txt")
            with open(report_path, "w") as f:
                f.write(f"Dataset base: {base}\n")
                f.write(f"Progression type: {progression_type}\n")
                f.write(f"MMSE imputation: {'post-split (train-fit)' if mmse_imputer_obj else 'not needed'}\n")
                f.write(f"CV method: {cv_method}\n")
                f.write(f"Class balancing: {'SMOTENC' if use_smote else 'disabled'}\n")
                f.write("Best hyperparameters:\n")
                for k, v in model.get_params().items():
                    if k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state", "objective", "eval_metric"]:
                        f.write(f"  {k}: {v}\n")
                f.write("\nClassification Report:\n")
                f.write(summary["classification_report"] + "\n")
                f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
                bm = summary["bootstrap_metrics"]
                f.write(f"Bootstrap 95% CI (n=1000, valid_samples={bm['valid_samples']}):\n")
                f.write(f"- Accuracy: {bm['accuracy'][0]:.3f} (CI: {bm['accuracy'][1][0]:.3f}, {bm['accuracy'][1][1]:.3f}) range={bm['accuracy'][1][1] - bm['accuracy'][1][0]:.3f}\n")
                f.write(f"- Precision (macro): {bm['precision_macro'][0]:.3f} (CI: {bm['precision_macro'][1][0]:.3f}, {bm['precision_macro'][1][1]:.3f}) range={bm['precision_macro'][1][1] - bm['precision_macro'][1][0]:.3f}\n")
                f.write(f"- Recall (macro): {bm['recall_macro'][0]:.3f} (CI: {bm['recall_macro'][1][0]:.3f}, {bm['recall_macro'][1][1]:.3f}) range={bm['recall_macro'][1][1] - bm['recall_macro'][1][0]:.3f}\n")
                f.write(f"- F1 (macro): {bm['f1_macro'][0]:.3f} (CI: {bm['f1_macro'][1][0]:.3f}, {bm['f1_macro'][1][1]:.3f}) range={bm['f1_macro'][1][1] - bm['f1_macro'][1][0]:.3f}\n")
                auc_lo, auc_hi = bm['auc'][1]
                if not np.isnan(auc_lo):
                    f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f}) range={auc_hi - auc_lo:.3f}\n")
                else:
                    f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI unavailable; too few valid resamples)\n")
            print(f"- Report:  {report_path}")

        return model, columns


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION  
# ═══════════════════════════════════════════════════════════════════════════════

def plot_feature_importance(importances, feature_names, top_n=20, title=None, save_path=None):
    """Publication-quality horizontal bar chart of top-N feature importances.

    Parameters
    ----------
    importances : array-like or fitted model
        Either a 1-D array of importance values or a fitted model with
        a `feature_importances_` attribute.
    feature_names : list[str]
        Feature names corresponding to the importance values.
    top_n : int
        Number of top features to display.
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    if hasattr(importances, 'feature_importances_'):
        importances = importances.feature_importances_
    fi = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi = fi.sort_values('Importance', ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(fi)))
    ax.barh(fi['Feature'], fi['Importance'], color=colors, edgecolor='white', linewidth=0.5)
    for i, (val, name) in enumerate(zip(fi['Importance'], fi['Feature'])):
        ax.text(val + fi['Importance'].max() * 0.01, i, f'{val:.4f}', va='center', fontsize=8)
    ax.set_xlabel('Importance (gain)', fontsize=12)
    ax.set_title(title or f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_mat(y_true, y_pred, class_labels=None, title=None, save_path=None):
    """Publication-quality annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    class_labels : list[str], optional
        Display names for classes (default: ['0', '1']).
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    cm = confusion_matrix(y_true, y_pred)
    if class_labels is None:
        class_labels = [str(c) for c in sorted(np.unique(y_true))]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels,
                yticklabels=class_labels, linewidths=0.5, linecolor='gray',
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title or 'Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc(y_true, y_proba, title=None, save_path=None):
    """Publication-quality ROC curve with AUC annotation and diagonal reference.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    title : str, optional
        Custom plot title.
    save_path : str, optional
        If provided, save the figure to this path.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2E86AB', lw=2.5, label=f'ROC curve (AUC = {auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2E86AB')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title or 'Receiver Operating Characteristic', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return fpr, tpr, auc_val


def plot_feature_correlation(df, figsize):
    """Correlation heatmap for an already-aggregated feature DataFrame."""
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    corr = df_imputed.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap='coolwarm',
        center=0,
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Map (Original Features)")
    plt.tight_layout()
    plt.show()


def eval_model(model, x, y):
    """Print classification report + ROC AUC for an already-fitted model."""
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)[:, 1]

    print("Classification Report:")
    print(classification_report(y, y_pred))
    print(f"\nROC AUC Score: {roc_auc_score(y, y_proba):.4f}")

    fpr, tpr, _ = roc_curve(y, y_proba)
    return fpr, tpr, roc_auc_score(y, y_proba)


# ═══════════════════════════════════════════════════════════════════════════════
#  LEAD-TIME ANALYSIS  
# ═══════════════════════════════════════════════════════════════════════════════

def add_time_dimension(df, months_between_visits=12):
    """Add a ``months_since_baseline`` column (list per patient) to *df*."""
    df = df.copy()
    df['months_since_baseline'] = df['Progression'].apply(
        lambda x: [i * months_between_visits for i in range(len(eval(x)))]
    )
    return df


def create_delta_features_truncated(df, max_visit):
    """Visit-agnostic features using only the first *max_visit* visits.

    Identical feature set to ``create_delta_features`` but truncates each
    time-series before engineering — used for lead-time analysis.
    The output column set is constant regardless of *max_visit*, enabling
    a single trained model to score any truncation length.
    """
    df = df.copy()
    new_columns = {}

    _parse_array_columns(df)

    # Truncate list columns to max_visit before engineering
    for col in df.columns:
        if _is_numeric_list_col(df[col]):
            df[col] = df[col].apply(
                lambda x: x[:max_visit] if isinstance(x, list) else np.nan)

    _engineer_visit_agnostic(df, new_columns)

    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    array_cols = [
        c for c in df.columns
        if isinstance(df[c].iloc[0], list) and c != 'months_since_baseline'
    ] if len(df) > 0 else []
    return df.drop(columns=array_cols)


def get_first_progression_visit(row, progression_code=1):
    """Return the 1-indexed visit number where progression first occurred, or NaN."""
    progression = eval(row['Progression'])
    for i, code in enumerate(progression):
        if code >= progression_code:
            return i + 1
    return np.nan


def evaluate_lead_time(df, model, imputer, scaler, feature_names,
                       threshold, progression_type='CN', min_visits=2):
    """
    For each progressor in *df*, find the earliest visit at which the
    (single, visit-agnostic) model exceeds *threshold* and compute the
    lead time (months before physician diagnosis).

    Parameters
    ----------
    df              : DataFrame with 'Progression' and 'months_since_baseline'.
    model           : trained XGBClassifier (visit-agnostic features).
    imputer, scaler : fitted SimpleImputer / StandardScaler.
    feature_names   : list[str] — feature columns the model was trained on.
    threshold       : probability threshold for a positive prediction.
    progression_type : 'CN' or 'AD'.
    min_visits      : minimum visits required to make a prediction (need ≥2
                      for slope / acceleration).

    Returns
    -------
    list of lead times in months.
    """
    lead_times = []
    prog_code = 2 if progression_type == 'AD' else 1

    for idx, row in df.iterrows():
        try:
            progression = eval(row['Progression'])
            months = row['months_since_baseline']
            if not isinstance(months, list):
                continue

            # Find physician diagnosis visit
            diag_idx = next(
                (i for i, c in enumerate(progression) if c >= prog_code), None)
            if diag_idx is None:
                continue
            diag_time = months[diag_idx]

            # Check model predictions at progressively more visits
            row_df = df.loc[[idx]]
            for visit in range(min_visits, diag_idx + 1):
                try:
                    trunc = create_delta_features_truncated(row_df, max_visit=visit)
                    # Align columns to training features
                    for col in feature_names:
                        if col not in trunc.columns:
                            trunc[col] = np.nan
                    X = trunc[feature_names].values
                    X_proc = scaler.transform(imputer.transform(X))
                    proba = model.predict_proba(X_proc)[0, 1]
                    if proba >= threshold:
                        lead_times.append(diag_time - months[visit - 1])
                        break
                except Exception as e:
                    print(f"  visit {visit}, patient {idx}: {e}")
                    continue
        except Exception as e:
            print(f"  patient {idx}: {e}")
            continue

    return lead_times


def evaluate_lead_time_full(df, model, imputer, scaler, feature_names,
                            threshold, progression_type='CN', min_visits=2):
    """
    Evaluate both progressors AND non-progressors on a lead-time cohort.

    For progressors: checks whether the model flags them before diagnosis
    and how early (lead time). For non-progressors: checks whether the
    model incorrectly flags them (false positives) using all their visits.

    Returns
    -------
    dict with keys:
        'lead_times'       : list[float] — months of early detection per TP
        'true_positives'   : int — progressors detected before diagnosis
        'false_negatives'  : int — progressors NOT detected
        'true_negatives'   : int — non-progressors correctly left alone
        'false_positives'  : int — non-progressors incorrectly flagged
        'total_progressors': int
        'total_non_progressors': int
        'sensitivity'      : float — TP / (TP + FN)
        'specificity'      : float — TN / (TN + FP)
        'fp_details'       : list[dict] — visit/proba info for each FP
    """
    prog_code = 2 if progression_type == 'AD' else 1

    lead_times = []
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    fp_details = []
    total_prog = 0
    total_nonprog = 0

    for idx, row in df.iterrows():
        try:
            progression = eval(row['Progression'])
            months = row.get('months_since_baseline')
            if not isinstance(months, list):
                continue

            diag_idx = next(
                (i for i, c in enumerate(progression) if c >= prog_code), None)
            is_progressor = diag_idx is not None

            if is_progressor:
                total_prog += 1
                diag_time = months[diag_idx]
                detected = False
                row_df = df.loc[[idx]]
                for visit in range(min_visits, diag_idx + 1):
                    try:
                        trunc = create_delta_features_truncated(row_df, max_visit=visit)
                        for col in feature_names:
                            if col not in trunc.columns:
                                trunc[col] = np.nan
                        X = trunc[feature_names].values
                        X_proc = scaler.transform(imputer.transform(X))
                        proba = model.predict_proba(X_proc)[0, 1]
                        if proba >= threshold:
                            lead_times.append(diag_time - months[visit - 1])
                            tp += 1
                            detected = True
                            break
                    except Exception:
                        continue
                if not detected:
                    fn += 1
            else:
                # Non-progressor: run through ALL visits and check for false alarm
                total_nonprog += 1
                flagged = False
                row_df = df.loc[[idx]]
                n_visits = len(progression)
                for visit in range(min_visits, n_visits + 1):
                    try:
                        trunc = create_delta_features_truncated(row_df, max_visit=visit)
                        for col in feature_names:
                            if col not in trunc.columns:
                                trunc[col] = np.nan
                        X = trunc[feature_names].values
                        X_proc = scaler.transform(imputer.transform(X))
                        proba = model.predict_proba(X_proc)[0, 1]
                        if proba >= threshold:
                            fp += 1
                            fp_details.append({
                                'patient_idx': idx,
                                'flagged_at_visit': visit,
                                'probability': round(float(proba), 3),
                            })
                            flagged = True
                            break
                    except Exception:
                        continue
                if not flagged:
                    tn += 1

        except Exception:
            continue

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None

    return {
        'lead_times': lead_times,
        'true_positives': tp,
        'false_negatives': fn,
        'true_negatives': tn,
        'false_positives': fp,
        'total_progressors': total_prog,
        'total_non_progressors': total_nonprog,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fp_details': fp_details,
    }
