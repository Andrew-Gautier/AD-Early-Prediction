import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
import os
import itertools
import warnings
import joblib
from joblib import Parallel, delayed
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
from preprocessing import create_target
from feature_engineering import create_delta_features, preprocess_data

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


def _fit_xgb(X_tr, X_te, y_tr, params, n_jobs=None, scale=False):
    """Fit (optional scaler +) XGBoost on training data; return predictions on test data.

    Parameters
    ----------
    scale : bool, default False
        When True, apply StandardScaler to X_tr / X_te before fitting.

    Returns (model, imputer, scaler, y_pred, y_proba).
    imputer is always None (reserved for future re-addition).
    scaler  is None when scale=False.
    """
    if scale:
        scaler = StandardScaler()
        X_tr_proc = scaler.fit_transform(X_tr)
        X_te_proc = scaler.transform(X_te)
    else:
        scaler = None
        X_tr_proc = X_tr
        X_te_proc = X_te

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        scale_pos_weight=spw,
        enable_categorical=True,
        n_jobs=n_jobs,
        random_state=42,
    )
    model.fit(X_tr_proc, y_tr)
    y_proba = model.predict_proba(X_te_proc)[:, 1]
    y_pred = model.predict(X_te_proc)
    return model, None, scaler, y_pred, y_proba


def _print_bootstrap_ci(metrics):
    """Print bootstrap 95% CI summary block to stdout."""
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


def _write_bootstrap_ci(f, bm):
    """Write bootstrap 95% CI summary block to an open file handle."""
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


def _run_loocv_folds(df, combo, progression_type, n_jobs, desc, tqdm_position=None):
    """Run all LOO folds for one combo; returns list of fold result tuples."""
    n_samples = len(df)
    pos_kwargs = {"position": tqdm_position} if tqdm_position is not None else {}
    if n_jobs == 1:
        return [
            _loocv_fold(i, df, combo, progression_type)
            for i in tqdm(range(n_samples), desc=desc, unit="fold", leave=False, **pos_kwargs)
        ]
    return list(tqdm(
        Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_loocv_fold)(i, df, combo, progression_type)
            for i in range(n_samples)
        ),
        total=n_samples,
        desc=desc,
        unit="fold",
        leave=False,
        **pos_kwargs,
    ))


# For the training of the best model with the best set of hyperparameters, prints out the whole performance report.
def build_model_final(X_train, X_test, y_train, y_test, model_dict, feature_names):
    
    model, imputer, scaler, y_pred, y_proba = _fit_xgb(X_train, X_test, y_train, model_dict)
    
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

    _print_bootstrap_ci(metrics)

    summary = {
        "classification_report": cr_str,
        "base_auc": float(base_auc),
        "bootstrap_metrics": metrics,
        "y_true": y_test,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "feature_names": list(feature_names),
        "feature_importances": model.feature_importances_,
    }
    return model, feature_names, imputer, scaler, summary

# To train models for cross-validation. Returns only AUC score. Does not print out anything.
def build_model(X_train, X_test, y_train, y_test, model_dict, feature_names, xgb_n_jobs=None):
    
    _, _, _, _, y_proba = _fit_xgb(X_train, X_test, y_train, model_dict, n_jobs=xgb_n_jobs)
    return roc_auc_score(y_test, y_proba)


def _loocv_fold(fold_idx, df_with_target, hyperparams, progression_type):
    """Run one LOO fold from raw data through prediction.

    The full pipeline (delta features -> preprocess -> impute/scale -> XGBoost)
    is executed per fold.

    Returns (true_label, predicted_proba, predicted_class).
    """
    warnings.filterwarnings('ignore')
    n_est, max_depth, lr, subsample, colsample = hyperparams

    # 1. train / test split
    all_idx = list(range(len(df_with_target)))
    train_idx = all_idx[:fold_idx] + all_idx[fold_idx + 1:]
    df_train = df_with_target.iloc[train_idx].copy()
    df_test  = df_with_target.iloc[[fold_idx]].copy()
    true_label = int(df_test['target'].iloc[0])

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

    # 7. Train + predict
    params = {
        'n_estimators': n_est, 'max_depth': max_depth, 'learning_rate': lr,
        'subsample': subsample, 'colsample_bytree': colsample,
    }
    clf, _, _, y_pred_arr, y_proba_arr = _fit_xgb(X_tr, X_te, y_tr, params, n_jobs=1)
    proba = float(y_proba_arr[0])
    pred  = int(y_pred_arr[0])

    return (true_label, proba, pred, feat_names, clf.feature_importances_)


def _eval_loocv_combo_pipeline(combo, df_with_target, progression_type,
                                n_jobs_folds=1, combo_label=None):
    """Evaluate one hyperparameter combo across all LOO folds (full pipeline per fold).
    Returns result tuple: (n_est, max_depth, lr, subsample, colsample, auc_score).

    n_jobs_folds : int
        Number of parallel workers for fold evaluation within this combo.
        1 = serial (default). -1 = use all available CPUs.
        Note: when this function is itself called inside an outer Parallel (combo-level),
        joblib will clamp this to 1 automatically to avoid over-subscription.
    combo_label : str or None
        Optional label for the tqdm fold progress bar (e.g. '3/36').
    """
    warnings.filterwarnings('ignore')
    fold_desc = f"Folds ({combo_label})" if combo_label else "Folds"
    results = _run_loocv_folds(df_with_target, combo, progression_type,
                               n_jobs=n_jobs_folds, desc=fold_desc, tqdm_position=1)
    y_true  = np.array([r[0] for r in results])
    y_score = np.array([r[1] for r in results])
    try:
        score = roc_auc_score(y_true, y_score)
    except Exception:
        score = 0.0
    return (*combo, score)


def _loocv_final_evaluation(df_with_target, best_params, progression_type,
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

    if n_jobs != 1:
        print(f"Running LOOCV final evaluation in parallel (n_jobs={n_jobs}, {n_samples} folds)...")
    results = _run_loocv_folds(df_with_target, combo, progression_type,
                               n_jobs=n_jobs, desc="LOOCV final evaluation")

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
    _print_bootstrap_ci(metrics)

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


def _eval_skf_combo(combo, list_x_train, list_x_test, list_y_train, list_y_test, n_splits, feature_names):
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
            xgb_n_jobs=1,
        )
    score = score_sum / n_splits
    return (n_estimators, max_depth, learning_rate, subsample, colsample_bytree, score)


# Perform grid search with the training dataset and the given parameter ranges.
# Return the set of best hyperparameters and save cross-validation scores to the given csv path.
# A helper method for train_best_model(...)
#
# cv_method    : 'skf'   → StratifiedKFold (default)
#                'loocv' → Leave-One-Out CV
# n_jobs       : number of parallel workers for evaluating hyperparameter combos (1 = serial)
# n_jobs_folds : number of parallel workers for LOO folds within each combo (1 = serial).
#                Ignored by the SKF branch. When n_jobs > 1 AND n_jobs_folds != 1 the two
#                axes run concurrently; joblib's loky backend clamps inner workers to avoid
#                runaway over-subscription, but prefer using only one axis at a time.
# n_jobs_folds : (LOOCV only) parallel workers for LOO folds within each grid-search combo and
#                the final LOOCV evaluation. 1 = serial (default). -1 = all CPUs.
def grid_search(x, y, param_grid, csv_path, feature_names, cv_method='skf', n_jobs=1,
                df_raw=None, progression_type=None,
                n_jobs_folds=1):
    best_hyperparameters = None
    best_score = 0

    combo_keys = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
    all_combos = list(itertools.product(*(param_grid[k] for k in combo_keys)))
    total_combos = len(all_combos)

    # ── LOOCV branch (full pipeline per fold) ────────────────────────────────────
    if cv_method == 'loocv':
        n_samples = len(df_raw)
        print(f"Using Leave-One-Out CV ({n_samples} folds x {total_combos} combos, "
              f"n_jobs(combos)={n_jobs}, n_jobs_folds={n_jobs_folds})")

        scores = []
        if n_jobs == 1:
            # Serial over combos — fold-level Parallel(n_jobs_folds) has full CPU access.
            # Using a plain loop avoids creating a joblib nesting context that would
            # silently clamp the inner Parallel to 1 worker.
            with tqdm(total=total_combos, desc="LOOCV grid search", unit="combo", position=0) as pbar:
                for idx, combo in enumerate(all_combos):
                    combo_label = f"{idx+1}/{total_combos}"
                    result = _eval_loocv_combo_pipeline(
                        combo, df_raw, progression_type,
                        n_jobs_folds=n_jobs_folds,
                        combo_label=combo_label,
                    )
                    scores.append(list(result))
                    if result[5] > best_score:
                        best_score = result[5]
                        best_hyperparameters = dict(zip(combo_keys, result[:5]))
                    pbar.update(1)
        else:
            # Parallel over combos — joblib's nesting guard auto-clamps any inner Parallel
            # to 1 worker, so n_jobs_folds is forced serial here.
            results = Parallel(n_jobs=n_jobs, return_as='generator')(
                delayed(_eval_loocv_combo_pipeline)(
                    combo, df_raw, progression_type,
                    n_jobs_folds=1,
                )
                for combo in all_combos
            )
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
            print(f"Note: Using StratifiedKFold with n_splits={n_splits} due to minority class size={min_class}")
        else:
            print(f"Using StratifiedKFold with n_splits={n_splits}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for i, (train_ind, test_ind) in enumerate(skf.split(x, y_arr)):
            list_x_train.append(x[train_ind])
            list_y_train.append(y_arr[train_ind])
            list_x_test.append(x[test_ind])
            list_y_test.append(y_arr[test_ind])

        print(f"Grid search: {total_combos} hyperparameter combinations (n_jobs={n_jobs})")

        results = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_eval_skf_combo)(combo, list_x_train, list_x_test, list_y_train, list_y_test, n_splits, feature_names)
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
#
# LOOCV modes:
#   loocv : no train/test split; LOOCV on full dataset IS the evaluation.
# n_jobs_folds : (LOOCV only) parallel workers for LOO folds within each grid-search combo and
#                the final LOOCV evaluation. 1 = serial (default). -1 = all CPUs.
#                Combine with n_jobs=1 for fold-level-only parallelism (recommended), or
#                leave at 1 and set n_jobs > 1 for combo-level-only parallelism.
def train_best_model(dataset, progression_type, param_grid, csv_path, save_dir="saved_models", model_base_name=None, save_artifacts=True, cv_method='skf', n_jobs=1, n_jobs_folds=1):
    
    dataset = dataset.copy()

    # --- Step 1: Create target variable on the raw DataFrame (needed for stratified split) ---
    dataset['target'] = dataset['Progression'].apply(create_target, progression_type=progression_type)

    # ══════════════════════════════════════════════════════════════════════════
    #  LOOCV mode (no train/test split)
    # ══════════════════════════════════════════════════════════════════════════
    if cv_method == 'loocv':
        print(f"\n{'='*60}")
        print(f"LOOCV — full dataset ({len(dataset)} samples)")
        print(f"{'='*60}")

        # Suppress noisy warnings (convergence, deprecation, etc.) during grid search
        warnings.filterwarnings('ignore')

        model_dict = grid_search(
            None, None, param_grid, csv_path, None,
            cv_method='loocv', n_jobs=n_jobs,
            df_raw=dataset,
            progression_type=progression_type,
            n_jobs_folds=n_jobs_folds,
        )

        # Final evaluation: re-run LOOCV with best hyperparameters
        summary = _loocv_final_evaluation(
            dataset, model_dict,
            progression_type=progression_type,
            n_jobs=n_jobs_folds,
        )

        # Save report only (LOOCV produces N models; no single model to persist)
        if save_artifacts:
            os.makedirs(save_dir, exist_ok=True)
            base = model_base_name or os.path.splitext(os.path.basename(csv_path))[0]
            report_path = os.path.join(save_dir, f"{base}_report_{progression_type}.txt")
            with open(report_path, "w") as f:
                f.write(f"Dataset base: {base}\n")
                f.write(f"Progression type: {progression_type}\n")
                f.write(f"CV method: loocv (no train/test split)\n")
                f.write(f"Total samples: {len(dataset)}\n")
                f.write("\nBest hyperparameters:\n")
                for k, v in summary['best_params'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\nClassification Report:\n")
                f.write(summary["classification_report"] + "\n")
                f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
                bm = summary["bootstrap_metrics"]
                _write_bootstrap_ci(f, bm)
            print(f"\nSaved report: {report_path}")

        return None, None, None  # No single model in LOOCV mode

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

        # --- Step 5: Grid search ---
        model_dict = grid_search(X_train, y_train, param_grid, csv_path, feature_names, cv_method=cv_method, n_jobs=n_jobs)

        # --- Step 6: Final model with full report ---
        model, columns, imputer, scaler, summary = build_model_final(X_train, X_test, y_train, y_test, model_dict, feature_names)

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

            report_path = os.path.join(save_dir, f"{base}_report_{progression_type}.txt")
            with open(report_path, "w") as f:
                f.write(f"Dataset base: {base}\n")
                f.write(f"Progression type: {progression_type}\n")
                f.write(f"CV method: {cv_method}\n")
                f.write("Best hyperparameters:\n")
                for k, v in model.get_params().items():
                    if k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state", "objective", "eval_metric"]:
                        f.write(f"  {k}: {v}\n")
                f.write("\nClassification Report:\n")
                f.write(summary["classification_report"] + "\n")
                f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
                bm = summary["bootstrap_metrics"]
                _write_bootstrap_ci(f, bm)
            print(f"- Report:  {report_path}")

        return model, columns, summary


# ══════════════════════════════════════════════════════════════════════════
#  Coarse grid search → freeze params → RFECV feature selection
# ══════════════════════════════════════════════════════════════════════════

def _encode_categoricals_for_rfe(df, categorical_cols):
    """Ordinal-encode category-dtype columns in *df* for use with sklearn RFECV.

    sklearn's RFECV strips pandas dtypes during internal column indexing, which
    would break XGBoost's enable_categorical=True.  We encode only the columns
    that are actually present and have category dtype.

    Returns
    -------
    df_encoded : pd.DataFrame
        Copy of *df* with category columns replaced by integer codes (-1 for NaN).
    encoding_map : dict  {col_name: {int_code: original_category, ...}}
        Mapping needed to restore category dtype after feature selection.
    """
    df_encoded = df.copy()
    encoding_map = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        if not hasattr(df[col], 'cat'):
            continue
        categories = df[col].cat.categories.tolist()
        encoding_map[col] = {i: cat for i, cat in enumerate(categories)}
        df_encoded[col] = df[col].cat.codes  # -1 for NaN
    return df_encoded, encoding_map


def _restore_categoricals(df, encoding_map):
    """Restore category dtype to columns previously encoded by _encode_categoricals_for_rfe.

    Only restores columns that survived feature selection (i.e. are still present in *df*).
    """
    df = df.copy()
    for col, code_to_cat in encoding_map.items():
        if col not in df.columns:
            continue
        n_cats = len(code_to_cat)
        categories = [code_to_cat[i] for i in range(n_cats)]
        # cat.codes uses -1 for NaN; convert back
        cat_series = pd.Categorical.from_codes(
            df[col].astype(int), categories=categories
        )
        df[col] = cat_series
    return df


def train_with_rfe(
    dataset,
    progression_type,
    param_grid,
    csv_path,
    save_dir="saved_models",
    model_base_name=None,
    save_artifacts=True,
    n_jobs=1,
    rfe_step=1,
    rfe_scoring="roc_auc",
    rfe_n_jobs=1,
):
    """Coarse SKF grid search → freeze best hyperparameters → RFECV feature selection → final model.

    Parameters
    ----------
    dataset         : pd.DataFrame — raw dataset with 'Progression' column.
    progression_type: str          — 'MCI', 'AD', or 'CN'.
    param_grid      : dict         — hyperparameter grid for the coarse grid search.
    csv_path        : str          — output path for grid search CV scores CSV.
    save_dir        : str          — directory for saved model / report.
    model_base_name : str | None   — base name for saved artifacts (inferred from csv_path if None).
    save_artifacts  : bool         — whether to save model + report to disk.
    n_jobs          : int          — parallel workers for the grid search combo loop.
    rfe_step        : int | float  — features removed per RFECV round (int) or fraction (float 0–1).
                                     Smaller = more thorough, slower.  Default 1.
    rfe_scoring     : str          — sklearn scoring metric for RFECV.  Default 'roc_auc'.
    rfe_n_jobs      : int          — parallel workers for RFECV CV folds.

    Returns
    -------
    model           : XGBClassifier trained on selected features.
    selected_features : list[str]  — feature names chosen by RFECV.
    rfecv           : fitted RFECV object (exposes .cv_results_, .n_features_, .support_).
    """
    from feature_engineering import create_delta_features, preprocess_data, CATEGORICAL_COLUMNS

    dataset = dataset.copy()

    # ── Step 1: target column ────────────────────────────────────────────────
    dataset['target'] = dataset['Progression'].apply(
        create_target, progression_type=progression_type
    )

    # ── Step 2: stratified 80/20 train/test split ────────────────────────────
    train_idx, test_idx = train_test_split(
        dataset.index, test_size=0.2, random_state=42, stratify=dataset['target']
    )
    df_train_raw = dataset.loc[train_idx].copy()
    df_test_raw  = dataset.loc[test_idx].copy()

    # ── Step 3: feature engineering + preprocessing ──────────────────────────
    y_train = df_train_raw['target'].values
    y_test  = df_test_raw['target'].values
    df_train_raw = df_train_raw.drop(columns=['target'])
    df_test_raw  = df_test_raw.drop(columns=['target'])

    df_train_feat = create_delta_features(df_train_raw)
    df_test_feat  = create_delta_features(df_test_raw)
    df_train_feat['target'] = y_train
    df_test_feat['target']  = y_test

    processed_train, _, _ = preprocess_data(df_train_feat, progression_type)
    processed_test,  _, _ = preprocess_data(df_test_feat,  progression_type)

    feature_names = [c for c in processed_train.columns if c != 'target']
    for col in feature_names:
        if col not in processed_test.columns:
            processed_test[col] = np.nan
    processed_test = processed_test[feature_names + ['target']]

    X_train_df = processed_train.drop(columns=['target'])
    X_test_df  = processed_test.drop(columns=['target'])
    X_train    = X_train_df.values
    X_test     = X_test_df.values

    # ── Step 4: coarse grid search with frozen SKF ───────────────────────────
    print(f"\n{'='*60}")
    print("Phase 1 — Coarse grid search (SKF)")
    print(f"{'='*60}")
    best_params = grid_search(
        X_train, y_train, param_grid, csv_path,
        feature_names, cv_method='skf', n_jobs=n_jobs,
    )
    print(f"\nBest hyperparameters: {best_params}")

    # ── Step 5: build frozen estimator (enable_categorical=False for RFECV) ──
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    frozen_estimator = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        scale_pos_weight=spw,
        enable_categorical=False,   # ordinal-encoded input; no category dtype
        n_jobs=1,
        random_state=42,
    )

    # ── Step 6: ordinal-encode categoricals for RFECV ────────────────────────
    X_train_enc, encoding_map = _encode_categoricals_for_rfe(
        X_train_df, CATEGORICAL_COLUMNS
    )
    X_test_enc, _ = _encode_categoricals_for_rfe(X_test_df, CATEGORICAL_COLUMNS)

    # ── Step 7: RFECV ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Phase 2 — RFECV (step={rfe_step}, scoring={rfe_scoring})")
    print(f"Starting features: {len(feature_names)}")
    print(f"{'='*60}")

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rfecv = RFECV(
        estimator=frozen_estimator,
        step=rfe_step,
        cv=cv_strategy,
        scoring=rfe_scoring,
        n_jobs=rfe_n_jobs,
        min_features_to_select=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rfecv.fit(X_train_enc.values, y_train)

    selected_features = [f for f, s in zip(feature_names, rfecv.support_) if s]
    print(f"\nOptimal features selected: {rfecv.n_features_} / {len(feature_names)}")
    print(f"Eliminated: {len(feature_names) - rfecv.n_features_} features")
    print(f"Selected: {selected_features}")

    # ── Step 8: restore category dtype on the filtered DataFrames ────────────
    X_train_sel = _restore_categoricals(X_train_df[selected_features], encoding_map)
    X_test_sel  = _restore_categoricals(X_test_df[selected_features],  encoding_map)

    # ── Step 9: final model on selected features ──────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 3 — Final model on selected features")
    print(f"{'='*60}")
    model, _, _, summary = None, None, None, None
    model, _, imputer, scaler, summary = build_model_final(
        X_train_sel.values, X_test_sel.values,
        y_train, y_test,
        best_params, selected_features,
    )

    # ── Step 10: save artifacts ───────────────────────────────────────────────
    if save_artifacts:
        os.makedirs(save_dir, exist_ok=True)
        base = model_base_name or os.path.splitext(os.path.basename(csv_path))[0]
        model_path  = os.path.join(save_dir, f"{base}_rfe_model_{progression_type}.pkl")
        report_path = os.path.join(save_dir, f"{base}_rfe_report_{progression_type}.txt")

        joblib.dump(model, model_path)

        with open(report_path, "w") as f:
            f.write(f"Dataset base: {base}\n")
            f.write(f"Progression type: {progression_type}\n")
            f.write(f"CV method: skf (coarse grid search) + RFECV\n")
            f.write(f"\nBest hyperparameters:\n")
            for k, v in best_params.items():
                f.write(f"  {k}: {v}\n")
            f.write(f"\nRFECV settings: step={rfe_step}, scoring={rfe_scoring}\n")
            f.write(f"Features before RFE: {len(feature_names)}\n")
            f.write(f"Features after RFE:  {rfecv.n_features_}\n")
            f.write(f"\nSelected features:\n")
            for feat in selected_features:
                f.write(f"  {feat}\n")
            f.write("\nClassification Report:\n")
            f.write(summary["classification_report"] + "\n")
            f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
            _write_bootstrap_ci(f, summary["bootstrap_metrics"])

        print(f"\nSaved artifacts:")
        print(f"- Model:  {model_path}")
        print(f"- Report: {report_path}")

    return model, selected_features, rfecv
