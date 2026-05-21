import numpy as np
import pandas as pd



# ═══════════════════════════════════════════════════════════════════════════════
#  LEAD-TIME ANALYSIS  
# ═══════════════════════════════════════════════════════════════════════════════

### Note: This will all need to be changed to the new lead time cohort structure/experimental design. 

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