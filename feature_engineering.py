import numpy as np
import pandas as pd
from scipy.stats import linregress




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