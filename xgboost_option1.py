import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from itertools import chain
from scipy.stats import linregress

df = pd.read_csv('Dataset_1//5visit_MCI_AD.csv')


def create_target_variable(row):
    id  = row['Prog_ID']
    return id

def create_delta_features(df):
    """
    Enhanced version that:
    1. Safely parses array strings first
    2. Handles mixed numeric/non-numeric data
    3. Creates features for variable visit lengths
    """
    df = df.copy()
    
    # First pass: Convert all array-strings to lists of floats
    for col in df.columns:
        if df[col].dtype == object and df[col].str.startswith('[').any():
            df[col] = df[col].apply(
                lambda x: [float(v.strip()) for v in x.replace('[','').replace(']','').split(',')] 
                if isinstance(x, str) and x.startswith('[') else np.nan
            )
    
    # Second pass: Create features only for numeric arrays
    for col in df.columns:
        if isinstance(df[col].iloc[0], list) and all(isinstance(v, (int, float)) for v in df[col].iloc[0] if not pd.isna(v)):
            max_visits = max(len(v) for v in df[col] if isinstance(v, list))
            
            # 1. Individual visit values
            for i in range(max_visits):
                df[f"{col}_V{i+1}"] = df[col].apply(
                    lambda x: x[i] if isinstance(x, list) and i < len(x) else np.nan
                )
            
            # 2. Deltas from baseline (V1)
            for i in range(1, max_visits):
                df[f"{col}_delta_V{i+1}-V1"] = df[col].apply(
                    lambda x: x[i]-x[0] if isinstance(x, list) and len(x) > i else np.nan
                )
            
            # 3. Slope of linear trend
            def calc_slope(x):
                if isinstance(x, list) and len(x) > 1:
                    x_vals = np.arange(len(x))
                    return linregress(x_vals, x).slope
                return np.nan
            df[f"{col}_slope"] = df[col].apply(calc_slope)
            
            # 4. Aggregates
            df[f"{col}_mean"] = df[col].apply(
                lambda x: np.mean(x) if isinstance(x, list) else np.nan
            )
            df[f"{col}_max"] = df[col].apply(
                lambda x: np.max(x) if isinstance(x, list) else np.nan
            )
    
    # Drop original array columns (keep only engineered features)
    array_cols = [col for col in df.columns if isinstance(df[col].iloc[0], list)]
    df = df.drop(columns=array_cols)
    
    return df

# Preprocess the data
def preprocess_data(df):
    # Create target variable
    df['target'] = df.apply(create_target_variable, axis=1)
    
    # Select features - adjust based on your domain knowledge
    features = [
        'SEX', 'EDUC', 'ALCOHOL', 'BMI', 'MMSE', 'GDS', 'CDR', 'TOBAC30',
        'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS',
        'PAYATTN', 'REMDATES', 'TRAVEL', 'NACCFAM', 'CVHATT', 'CVAFIB',
        'DIABETES', 'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 'ANX', 'NACCTBI',
        'SMOKYRS', 'RACE', 'age'
    ]
    
    # # Extract first visit data (simplifying for initial model)
    # for col in df.columns:
    #     if col in features and df[col].dtype == object:
    #         df[col] = df[col].apply(extract_first_value)
    
    # Handle missing values
    df = df[features + ['target']].copy()
    df = df.dropna(subset=features, how='all')
    
    # Convert categorical variables
    categorical_cols = ['SEX', 'NACCFAM', 'CVHATT', 'CVAFIB', 'DIABETES', 
                       'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 'ANX', 'NACCTBI', 'RACE']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

df_processed = create_delta_features(df)
processed_df = preprocess_data(df_processed)

# Check class distribution
print("Class distribution:")
print(df_processed['target'].value_counts())

def build_model(data):
    # Separate features and target
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train XGBoost model
    model2 = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)  # Handle class imbalance
    )
    
    model2.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model2.predict(X_test)
    y_proba = model2.predict_proba(X_test)[:, 1]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    return model2, X.columns

# Build and evaluate the model
model2, feature_names = build_model(df_imputed)

def plot_feature_importance(model, feature_names):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.show()

# Plot feature importance
plot_feature_importance(model2, feature_names)