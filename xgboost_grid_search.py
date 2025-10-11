import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from pauc import ROC, ci_auc

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
            new_columns[f"{col}_mean"] = df[col].apply(
                lambda x: np.nanmean(x) if isinstance(x, list) else np.nan
            )
            new_columns[f"{col}_max"] = df[col].apply(
                lambda x: np.nanmax(x) if isinstance(x, list) else np.nan
            )
    
    # Add all new columns to the DataFrame at once
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    
    # Drop original array columns (keep only engineered features)
    array_cols = [col for col in df.columns if isinstance(df[col].iloc[0], list)] if len(df) > 0 else []
    df = df.drop(columns=array_cols)
    
    return df

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
    
    X = df[features]
    
    # Fit scaler and imputer
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Update DataFrame with scaled values
    df_scaled = df.copy()
    df_scaled[features] = X_scaled
    
    # Return both the processed DataFrame AND the scaler
    return df_scaled, scaler, imputer


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

    # Fit scaler and imputer
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    
    # Recalculate all scalers and imputers for the training set
    X_train = scaler.fit_transform(imputer.fit_transform(X_train))

    return X_train, X_test, y_train, y_test

# For the training of the best model with the best set of hyperparameters, prints out the whole performance report.
def build_model_final(X_train, X_test, y_train, y_test, model_dict):
    
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
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    roc_obj = ROC(y_true= y_test, y_score= y_proba)
    # Calculate AUC and 95% confidence interval
    (lower_ci, upper_ci) = ci_auc(
        roc_obj,
        method="bootstrap",
        conf_level=0.95,
        n_boot=100,
        bounds=(0.6, 1.0),
        focus ="sensitivity"
    )

    print(f"Partial AUC (boostrap): {auc_value:.3f}")
    print(f"95% Confidence Interval: ({lower_ci:.3f}, {upper_ci:.3f})")

    return model, X_train.columns

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
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_proba = model.predict_proba(X_test)[:, 1]

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
    skf = StratifiedKFold(n_splits=5)
    for i, (train_ind, test_ind) in enumerate(skf.split(x,y)):
        # store 5 folds of training and testing dataset in lists
        list_x_train.append(x[train_ind])
        y_np = np.array(y)
        list_y_train.append(y_np[train_ind])
        list_x_test.append(x[test_ind])
        list_y_test.append(y_np[test_ind])

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
                        # train 5 models using 5 datasets from stratified k-fold
                        for i in range(5):
                            score_sum += build_model(list_x_train[i], list_x_test[i], list_y_train[i], list_y_test[i], model_dict)

                        # record and update score
                        score = score_sum / 5
                        scores.append([n_estimators, max_depth, learning_rate, subsample, colsample_bytree, score])

                        if (score > best_score):
                            best_score = score
                            best_hyperparameters = model_dict
                        
    # save the cross validations scores as cvs to the given path
    pd.DataFrame(scores, columns=["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "avg AUC score"]).to_csv(csv_path, index=False)
    return best_hyperparameters

# Input an unprocessed dataset, progression type for preprocessing, and the parameter ranges 
#   to get a best performing model with the best set of hyperparameters found from grid-search.
# Save cross-validation scores to the input new csv path.
def train_best_model(dataset, progression_type, param_grid, csv_path):
    processed_df, scaler, imputer = preprocess_data(create_delta_features(dataset), progression_type)
    
    # first dataset splitting
    X_train, X_test, y_train, y_test = split_dataset(processed_df)

    # Get the best set of hyperparameter from grid search, and save a cross-validation scores csv to the given path
    model_dict = grid_search(X_train, y_train, param_grid, csv_path)

    # print out a full report
    model, columns = build_model_final(X_train, X_test, y_train, y_test, model_dict)

    return model, columns


# MAIN LINES
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

    # Aptly call train_best_model function to start grid search. 
    # Be sure to save the returned model every time you call with a specific dataset.
    # Need a loop to go through all datatsets for diff progression types and visit counts and to generate corresponding csv paths. 


