import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
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
import joblib
from imblearn.over_sampling import SMOTENC

# This is the classifer used for visits 2-6 classification scores and charts. 

def create_target_variable_ad(row):
    """Create binary target from Progression column (1=progressed, 0=stable)"""
    progression = eval(row['Progression'])  # Convert string tuple to actual tuple
    return 1 if 2 in progression else 0  # 1 if progressed to AD at any visit
def create_target_variable_mci(row):
    """Create binary target from Progression column (1=progressed, 0=stable)"""
    progression = eval(row['Progression'])  # Convert string tuple to actual tuple
    return 1 if 1 in progression else 0  # 1 if progressed to AD at any visit
def create_delta_features(df):
    """
    Enhanced version that:
    1. Safely parses array strings first
    2. Handles mixed numeric/non-numeric data
    3. Creates features for variable visit lengths
    """
    df = df.copy()
    new_columns = {}  # Dictionary to store new columns

    # First pass: Convert all array-strings to lists of floats
    for col in df.columns:
        if df[col].dtype == object and df[col].str.startswith('[').any():
            df[col] = df[col].apply(
                lambda x: [float(v.strip()) if v.strip() != 'nan' else np.nan 
                         for v in x.replace('[','').replace(']','').split(',')] 
                if isinstance(x, str) and x.startswith('[') else np.nan
            )
    
    # Second pass: Create features only for numeric arrays
    for col in df.columns:
        if isinstance(df[col].iloc[0], list) and all(isinstance(v, (int, float)) for v in df[col].iloc[0] if not pd.isna(v)):
            max_visits = max(len(v) for v in df[col] if isinstance(v, list))
            
            # 1. Individual visit values
            for i in range(max_visits):
                new_columns[f"{col}_V{i+1}"] = df[col].apply(
                    lambda x: x[i] if isinstance(x, list) and i < len(x) else np.nan
                )
            
            # 2. Deltas from baseline (V1)
            for i in range(1, max_visits):
                new_columns[f"{col}_delta_V{i+1}-V1"] = df[col].apply(
                    lambda x: x[i]-x[0] if isinstance(x, list) and len(x) > i else np.nan
                )
            
            # 3. Slope of linear trend
            def calc_slope(x):
                if isinstance(x, list) and len(x) > 1:
                    x_vals = np.arange(len(x))
                    y_vals = np.array(x)
                    valid = ~np.isnan(y_vals)
                    if sum(valid) > 1:  # Need at least 2 points
                        return linregress(x_vals[valid], y_vals[valid]).slope
                return np.nan
                
            new_columns[f"{col}_slope"] = df[col].apply(calc_slope)
            
            # 4. Aggregates
            # Guard against empty/all-NaN lists to avoid warnings
            new_columns[f"{col}_mean"] = df[col].apply(
                lambda x: (np.nanmean(x)
                           if isinstance(x, list) and np.any(~np.isnan(x))
                           else np.nan)
            )
            new_columns[f"{col}_max"] = df[col].apply(
                lambda x: (np.nanmax(x)
                           if isinstance(x, list) and np.any(~np.isnan(x))
                           else np.nan)
            )
    
    # Add all new columns to the DataFrame at once
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
    min_valid_auc=30,
    verbose=True,
):
    """Bootstrap CIs for accuracy, precision (macro), recall (macro), F1 (macro), and ROC AUC.

    For AUC, resamples missing a class are skipped. Returns a dict:
      {
        'accuracy': (point, (lo, hi)),
        'precision_macro': (point, (lo, hi)),
        'recall_macro': (point, (lo, hi)),
        'f1_macro': (point, (lo, hi)),
        'auc': (point, (lo, hi))
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
    skipped_auc = 0
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        bt_y = y_true[idx]
        bt_pred = y_pred[idx]
        bt_score = y_score[idx]

        acc_b.append(accuracy_score(bt_y, bt_pred))
        prec_b.append(precision_score(bt_y, bt_pred, average="macro", zero_division=0))
        rec_b.append(recall_score(bt_y, bt_pred, average="macro", zero_division=0))
        f1_b.append(f1_score(bt_y, bt_pred, average="macro", zero_division=0))

        if len(np.unique(bt_y)) < 2:
            skipped_auc += 1
        else:
            try:
                auc_b.append(roc_auc_score(bt_y, bt_score))
            except ValueError:
                skipped_auc += 1

    if verbose:
        print(f"Bootstrap classification metrics: attempted={n_boot}, auc_valid={len(auc_b)}, auc_skipped={skipped_auc}")

    # Helper to CI from list
    def ci_bounds(vals):
        arr = np.array(vals)
        return (
            float(np.percentile(arr, 100 * (alpha / 2))),
            float(np.percentile(arr, 100 * (1 - alpha / 2))),
        )

    results = {
        "accuracy": (acc_pt, ci_bounds(acc_b)),
        "precision_macro": (prec_pt, ci_bounds(prec_b)),
        "recall_macro": (rec_pt, ci_bounds(rec_b)),
        "f1_macro": (f1_pt, ci_bounds(f1_b)),
        "auc": (auc_pt, (np.nan, np.nan)),
    }

    if len(auc_b) >= min_valid_auc:
        results["auc"] = (auc_pt, ci_bounds(auc_b))
    else:
        if verbose:
            print(f"Insufficient valid bootstrap samples for AUC (<{min_valid_auc}); CI unavailable.")

    return results

# Preprocess the data
def preprocess_data(df, progression_type):
    # Create target variable
    if progression_type == 'AD':
        df['target'] = df.apply(create_target_variable_ad, axis=1)
    elif progression_type == 'MCI':
        df['target'] = df.apply(create_target_variable_mci, axis=1)
    
    # Get all available features (after create_delta_features transformation)
    all_features = [col for col in df.columns if col != 'target']
    
    # Select features we want to keep
    # Static features (non-time-series)
    static_features = [
        'SEX', 'EDUC', 'ALCOHOL', 'NACCFAM', 'CVHATT', 
        'CVAFIB', 'DIABETES', 'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 
        'ANX', 'NACCTBI', 'SMOKYRS', 'RACE', 'age', 'HISPANIC'
    ]
    
    # Time-series features (we'll keep all of them)
    time_series_features = [col for col in all_features if '_V' in col or '_delta_' in col or '_slope' in col or '_mean' in col or '_max' in col]
    
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
def build_model_final(X_train, X_test, y_train, y_test, model_dict, feature_names):
    
    # Handle class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1

    # Build a model with the input hyperparameter values
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',

        n_estimators=model_dict['n_estimators'],
        max_depth=model_dict['max_depth'],
        learning_rate=model_dict['learning_rate'],
        subsample=model_dict['subsample'],
        colsample_bytree=model_dict['colsample_bytree'],

        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    # Fit imputer/scaler on training set only (avoid leakage)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_proc = scaler.transform(imputer.transform(X_test))

    model.fit(X_train_proc, y_train)
    
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
        min_valid_auc=30,
        verbose=True,
    )

    acc_pt, (acc_lo, acc_hi) = metrics["accuracy"]
    prec_pt, (prec_lo, prec_hi) = metrics["precision_macro"]
    rec_pt, (rec_lo, rec_hi) = metrics["recall_macro"]
    f1_pt, (f1_lo, f1_hi) = metrics["f1_macro"]
    auc_pt, (auc_lo, auc_hi) = metrics["auc"]

    print("\nBootstrap 95% CI (n=1000):")
    print(f"- Accuracy: {acc_pt:.3f} (CI: {acc_lo:.3f}, {acc_hi:.3f})")
    print(f"- Precision (macro): {prec_pt:.3f} (CI: {prec_lo:.3f}, {prec_hi:.3f})")
    print(f"- Recall (macro): {rec_pt:.3f} (CI: {rec_lo:.3f}, {rec_hi:.3f})")
    print(f"- F1 (macro): {f1_pt:.3f} (CI: {f1_lo:.3f}, {f1_hi:.3f})")
    if not np.isnan(auc_lo):
        print(f"- ROC AUC: {auc_pt:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f})")
    else:
        print(f"- ROC AUC: {auc_pt:.3f} (CI unavailable; too few valid resamples)")

    summary = {
        "classification_report": cr_str,
        "base_auc": float(base_auc),
        "bootstrap_metrics": metrics,
    }
    return model, feature_names, imputer, scaler, summary

# To train models for cross-validation. Returns only AUC score. Does not print out anything.
def build_model(X_train, X_test, y_train, y_test, model_dict):
    
    # Handle class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1

    # Build a model with the input hyperparameter values
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',

        n_estimators=model_dict['n_estimators'],
        max_depth=model_dict['max_depth'],
        learning_rate=model_dict['learning_rate'],
        subsample=model_dict['subsample'],
        colsample_bytree=model_dict['colsample_bytree'],

        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
    
    # Per-fold imputation/scaling fit on training fold only (avoid leakage)
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_proc = scaler.transform(imputer.transform(X_test))

    model.fit(X_train_proc, y_train)
    
    # Evaluate model
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    return roc_auc_score(y_test, y_proba)

# Perform grid search with the training dataset and the given parameter ranges.
# Return the set of best hyperparameters and save cross-validation scores to the given csv path.
# A helper method for train_best_model(...)
def grid_search(x, y, param_grid, csv_path):
    # a list to store hyperparameter values and cross validation scores
    scores = []

    best_hyperparameters = None
    best_score = 0

    list_x_train = []
    list_x_test = []
    list_y_train = []
    list_y_test = []
    # Determine feasible n_splits based on minority class count
    y_arr = np.array(y)
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
        # store 5 folds of training and testing dataset in lists
        list_x_train.append(x[train_ind])
        list_y_train.append(y_arr[train_ind])
        list_x_test.append(x[test_ind])
        list_y_test.append(y_arr[test_ind])

    for n_estimators in param_grid['n_estimators']:
        for max_depth in param_grid['max_depth']:
            for learning_rate in param_grid['learning_rate']:
                for subsample in param_grid['subsample']:
                    for colsample_bytree in param_grid['colsample_bytree']:
                        score_sum = 0
                        model_dict = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'subsample': subsample,
                            'colsample_bytree': colsample_bytree
                        }
                        # Train across folds
                        for i in range(n_splits):
                            score_sum += build_model(list_x_train[i], list_x_test[i], list_y_train[i], list_y_test[i], model_dict)

                        # record and update score
                        score = score_sum / n_splits
                        scores.append([n_estimators, max_depth, learning_rate, subsample, colsample_bytree, score])

                        if (score > best_score):
                            best_score = score
                            best_hyperparameters = model_dict
                        
    # ensure directory exists and save the cross validations scores as csv
    dirn = os.path.dirname(csv_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    pd.DataFrame(scores, columns=["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "avg AUC score"]).to_csv(csv_path, index=False)
    return best_hyperparameters

# Input an unprocessed dataset, progression type for preprocessing, and the parameter ranges 
#   to get a best performing model with the best set of hyperparameters found from grid-search.
# Save cross-validation scores to the input new csv path.
def train_best_model(dataset, progression_type, param_grid, csv_path, save_dir="saved_models", model_base_name=None, save_artifacts=True):
    processed_df, scaler, imputer = preprocess_data(create_delta_features(dataset), progression_type)
    feature_names = processed_df.drop(columns=['target']).columns.tolist()

    # first dataset splitting
    X_train, X_test, y_train, y_test = split_dataset(processed_df)

    # Get the best set of hyperparameter from grid search, and save a cross-validation scores csv to the given path
    model_dict = grid_search(X_train, y_train, param_grid, csv_path)

    # print out a full report
    model, columns, imputer, scaler, summary = build_model_final(X_train, X_test, y_train, y_test, model_dict, feature_names)

    # Attach feature names for downstream use (e.g., lead time notebook)
    try:
        model.feature_names_in_ = np.array(columns)
    except Exception:
        pass

    # Save artifacts if requested
    if save_artifacts:
        os.makedirs(save_dir, exist_ok=True)
        # Derive base name from provided model_base_name or csv_path stem
        if model_base_name is None:
            base = os.path.splitext(os.path.basename(csv_path))[0]
        else:
            base = model_base_name
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

        # Save a textual report
        report_path = os.path.join(save_dir, f"{base}_report_{progression_type}.txt")
        with open(report_path, "w") as f:
            f.write(f"Dataset base: {base}\n")
            f.write(f"Progression type: {progression_type}\n")
            f.write("Best hyperparameters:\n")
            for k, v in model.get_params().items():
                if k in ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "scale_pos_weight", "random_state", "objective", "eval_metric"]:
                    f.write(f"  {k}: {v}\n")
            f.write("\nClassification Report:\n")
            f.write(summary["classification_report"] + "\n")
            f.write(f"\nBase ROC AUC: {summary['base_auc']:.4f}\n")
            bm = summary["bootstrap_metrics"]
            f.write("Bootstrap 95% CI (n=1000):\n")
            f.write(f"- Accuracy: {bm['accuracy'][0]:.3f} (CI: {bm['accuracy'][1][0]:.3f}, {bm['accuracy'][1][1]:.3f})\n")
            f.write(f"- Precision (macro): {bm['precision_macro'][0]:.3f} (CI: {bm['precision_macro'][1][0]:.3f}, {bm['precision_macro'][1][1]:.3f})\n")\
            
            f.write(f"- Recall (macro): {bm['recall_macro'][0]:.3f} (CI: {bm['recall_macro'][1][0]:.3f}, {bm['recall_macro'][1][1]:.3f})\n")
            f.write(f"- F1 (macro): {bm['f1_macro'][0]:.3f} (CI: {bm['f1_macro'][1][0]:.3f}, {bm['f1_macro'][1][1]:.3f})\n")
            auc_lo, auc_hi = bm['auc'][1]
            if not np.isnan(auc_lo):
                f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI: {auc_lo:.3f}, {auc_hi:.3f})\n")
            else:
                f.write(f"- ROC AUC: {bm['auc'][0]:.3f} (CI unavailable; too few valid resamples)\n")
        print(f"- Report:  {report_path}")

    return model, columns

