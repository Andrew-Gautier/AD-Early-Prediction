import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns
from pauc import compute_auc_ci

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
    y = df['target']
    
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

def build_model(X_train, X_test, y_train, y_test, model_dict):
    
    # Handle class imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    
    # Train XGBoost model
    # model = XGBClassifier(
    #     objective='binary:logistic',
    #     eval_metric='auc',
    #     n_estimators=200,
    #     max_depth=5,
    #     learning_rate=0.1,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42,
    #     scale_pos_weight=scale_pos_weight
    # )

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

    # Calculate AUC and 95% confidence interval
    auc, lower_ci, upper_ci = compute_auc_ci(y_test, y_proba, alpha=0.05)

    print(f"AUC: {auc:.3f}")
    print(f"95% Confidence Interval: ({lower_ci:.3f}, {upper_ci:.3f})")

    return model, X_train.columns

def plot_feature_importance(model, feature_names, top_n=20):
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()
    
def plot_feature_correlation(df, figsize):
    """Plot correlation matrix for aggregated features"""
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Compute correlations
    corr = df_imputed.corr()
    
    # Plot heatmap
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
    # Evaluate model
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y, y_pred))
    
    print(f"\nROC AUC Score: {roc_auc_score(y, y_proba):.4f}")

    # Calculate AUC and 95% confidence interval
    auc, lower_ci, upper_ci = compute_auc_ci(y, y_proba, alpha=0.05)

    print(f"AUC: {auc:.3f}")
    print(f"95% Confidence Interval: ({lower_ci:.3f}, {upper_ci:.3f})")