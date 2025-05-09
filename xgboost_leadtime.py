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
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

def add_time_dimension(df, months_between_visits=12):
    """Adds a column for months elapsed since baseline (V1) for each visit."""
    df = df.copy()
    
    # Example: For a patient with 3 visits, time points are [0, 12, 24] months
    df['months_since_baseline'] = df['Progression'].apply(
        lambda x: [i * months_between_visits for i in range(len(eval(x)))]
    )
    return df

def create_target_variable_ad(row):
    """Create binary target from Progression column (1=progressed, 0=stable)"""
    progression = eval(row['Progression'])  # Convert string tuple to actual tuple
    return 1 if 2 in progression else 0  # 1 if progressed to AD at any visit
def create_target_variable_mci(row):
    """Create binary target from Progression column (1=progressed, 0=stable)"""
    progression = eval(row['Progression'])  # Convert string tuple to actual tuple
    return 1 if 1 in progression else 0  # 1 if progressed to AD at any visit

def create_delta_features_truncated(df, max_visit):
    """
    Create features up to a specified visit (e.g., max_visit=3 for visits 1-3).
    Preserves the months_since_baseline column and handles data truncation properly.
    """
    df = df.copy()
    new_columns = {}
    
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
            # Truncate to max_visit first
            df[col] = df[col].apply(lambda x: x[:max_visit] if isinstance(x, list) else np.nan)
            
            max_visits = min(max_visit, max(len(v) for v in df[col] if isinstance(v, list)))
            
            # 1. Individual visit values (only up to max_visit)
            for i in range(max_visits):
                new_columns[f"{col}_V{i+1}"] = df[col].apply(
                    lambda x: x[i] if isinstance(x, list) and i < len(x) else np.nan
                )
            
            # 2. Deltas from baseline (V1) - only if we have at least 2 visits
            if max_visits > 1:
                for i in range(1, max_visits):
                    new_columns[f"{col}_delta_V{i+1}-V1"] = df[col].apply(
                        lambda x: x[i]-x[0] if isinstance(x, list) and len(x) > i else np.nan
                    )
            
            # 3. Slope of linear trend (needs at least 2 visits)
            def calc_slope(x):
                if isinstance(x, list) and len(x) > 1:
                    x_vals = np.arange(len(x))
                    y_vals = np.array(x)
                    valid = ~np.isnan(y_vals)
                    if sum(valid) > 1:  # Need at least 2 points
                        return linregress(x_vals[valid], y_vals[valid]).slope
                return np.nan
                
            if max_visits > 1:
                new_columns[f"{col}_slope"] = df[col].apply(calc_slope)
            
            # 4. Aggregates (calculated on truncated visits)
            new_columns[f"{col}_mean"] = df[col].apply(
                lambda x: np.nanmean(x) if isinstance(x, list) else np.nan
            )
            new_columns[f"{col}_max"] = df[col].apply(
                lambda x: np.nanmax(x) if isinstance(x, list) else np.nan
            )
    
    # Add all new columns to the DataFrame at once
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
    
    # Drop original array columns except months_since_baseline
    array_cols = [
        col for col in df.columns
        if isinstance(df[col].iloc[0], list) and col != 'months_since_baseline'
    ] if len(df) > 0 else []
    
    if array_cols:
        df = df.drop(columns=array_cols)
    
    return df

def get_first_progression_visit(row, progression_code=1):
    """Return the first visit where progression to MCI (1) or AD (2) occurred."""
    progression = eval(row['Progression'])
    for i, code in enumerate(progression):
        if code >= progression_code:  # 1 for MCI, 2 for AD
            return i + 1  # Visits are 1-indexed
    return np.nan  # No progression

def evaluate_lead_time(df, models_dict, threshold, progression_type='MCI'):
    lead_times = []
    
    for idx, row in df.iterrows():
        try:
            progression = eval(row['Progression'])
            months_since_baseline = row['months_since_baseline']
            
            if not isinstance(months_since_baseline, list):
                continue
            
            # Find diagnosis time
            diagnosis_visit_idx = None
            for i, code in enumerate(progression):
                if code >= (2 if progression_type == 'AD' else 1):
                    diagnosis_visit_idx = i
                    break
            
            if diagnosis_visit_idx is None:
                continue
                
            diagnosis_time = months_since_baseline[diagnosis_visit_idx]
            
            # Check predictions at earlier visits
            for visit in range(1, diagnosis_visit_idx + 1):
                if visit not in models_dict:
                    continue
                    
                model_data = models_dict[visit]
                df_truncated = create_delta_features_truncated(df, max_visit=visit)
                
                try:
                    # Get just this patient's data
                    X = df_truncated.loc[[idx], model_data['model'].feature_names_in_]
                    
                    # Impute and scale
                    X_imputed = model_data['imputer'].transform(X)
                    X_scaled = model_data['scaler'].transform(X_imputed)
                    
                    # Get single probability
                    proba = model_data['model'].predict_proba(X_scaled)[0, 1]
                    
                    if proba >= threshold:
                        prediction_time = months_since_baseline[visit - 1]
                        lead_time = diagnosis_time - prediction_time
                        lead_times.append(lead_time)
                        break
                        
                except Exception as e:
                    print(f"Error processing visit {visit} for patient {idx}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing patient {idx}: {str(e)}")
            continue
    
    return lead_times

# def evaluate_lead_time(df, models_dict, progression_type='MCI', threshold=0.5):
#     lead_times = []
#     preprocessed = preprocess_data(df, progression_type)
#     scaler = preprocessed['scaler']
#     imputer = preprocessed['imputer']
#     features = preprocessed['features']
    
#     for _, row in df.iterrows():
#         progression = eval(row['Progression'])
#         months_since_baseline = row['months_since_baseline']  # Added in Step 1
        
#         # Skip if no progression or time data is missing
#         if not isinstance(months_since_baseline, list) or len(months_since_baseline) == 0:
#             continue
        
#         # Find physician's diagnosis time (first progression visit)
#         diagnosis_visit_idx = None
#         for i, code in enumerate(progression):
#             if code >= (2 if progression_type == 'AD' else 1):
#                 diagnosis_visit_idx = i
#                 break
        
#         if diagnosis_visit_idx is None:
#             continue  # No progression
        
#         diagnosis_time = months_since_baseline[diagnosis_visit_idx]
        
        
#         # Check model predictions at earlier visits
#         for visit in range(1, diagnosis_visit_idx + 1):
#             if visit not in models_dict:
#                 continue  # Skip if no model for this visit
                
#             model_data = models_dict[visit]
#             df_truncated = create_delta_features_truncated(df, max_visit=visit)
            
#             # Prepare features (ensure they match training features)
#             X = df_truncated[model_data['model'].feature_names_in_].iloc[[row.name]]  # Use XGBoost's built-in feature names
            
#             # Impute and scale
#             X_imputed = model_data['imputer'].transform(X)
#             X_scaled = model_data['scaler'].transform(X_imputed)
            
#             # Predict
#             proba = model_data['model'].predict_proba(X_scaled)[0, 1]  # Get probability for this specific sample
#             if proba >= threshold:
#                 prediction_time = months_since_baseline[visit - 1]
#                 lead_times.append(diagnosis_time - prediction_time)
#                 break  # Earliest prediction
    
#     return lead_times

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

    return df
