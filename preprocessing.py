import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
import json
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

def progressor_class(file_path):
    
    df = pd.read_csv(file_path)
    if 'Prog_ID' in df.columns:
        progressors = df['Prog_ID'] == 1
        non_progressors = df['Prog_ID'] == 0
        print(f"File: {file_path} - Progressors: {progressors.sum()}, Non-Progressors: {non_progressors.sum()}")

    mci_count = 0
    cn_to_mci = 0
    mci_to_ad = 0
    reverters = 0

    for ids, row in df.iterrows():
        progression = row['Progression']
        if isinstance(progression, str):
            progression = eval(progression)

        prog_id = row['Prog_ID']

        # Reverter check: progressor whose final visit reverts to their starting label
        # CN/MCI reverter: starts with 0, labeled progressor, ends with 0
        # MCI/AD reverter: starts with 1, labeled progressor, ends with 1
        is_reverter = (
            prog_id == 1 and (
                (progression[0] == 0 and progression[-1] == 0) or
                (progression[0] == 1 and progression[-1] == 1)
            )
        )
        if is_reverter:
            reverters += 1
            continue  # exclude from counts below

        if all(val == 1 for val in progression):
            mci_count += 1

        # CN to MCI: starts CN (0), progresses to MCI (1)
        if prog_id == 1 and progression[0] == 0 and 1 in progression:
            cn_to_mci += 1

        # MCI to AD: starts MCI (1), progresses to AD (2)
        if prog_id == 1 and progression[0] == 1 and 2 in progression:
            mci_to_ad += 1

    print(f"Reverters excluded in {file_path}: {reverters}")
    print(f"Stable MCI in {file_path}:          {mci_count}")
    print(f"Stable CN in {file_path}:           {non_progressors.sum() - mci_count}")
    print(f"CN to MCI in {file_path}:           {cn_to_mci}")
    print(f"MCI to AD in {file_path}:           {mci_to_ad}")

    return reverters
CODINGS = {
    "standard": {
        "name": "standard",
        "rules": [
            # (label, condition_fn)  — evaluated top-to-bottom, first match wins
            ("CN",      lambda row: row["NORMCOG"] == 1),
            ("AD",      lambda row: row["NACCALZP"] in (1, 2)),
            ("MCI",     lambda row: row["NACCUDSD"] == 3),
            ("Unknown", lambda row: True),   # catch-all
        ]
    },
    "cn_mci_dementia": {
        # Alternative: allows for
        "name": "cn_mci_dementia",
        "rules": [
            ("CN",      lambda row: row["NORMCOG"] == 1),
            ("MCI",      lambda row: row["NACCUDSD"] == 3),
            ("DEM",     lambda row: row["NACCUDSD"] == 4),
            ("Unknown", lambda row: True),
        ]
    },

}

# ── 2. Core labelling function ─────────────────────────────────────────────
def label_visit(row: pd.Series, rules: list) -> str:
    """Apply ordered rules to a single visit row, return first matching label."""
    for label, condition in rules:
        if condition(row):
            return label
    return "Unknown"

# ── 3. Grouping function — coding-agnostic ─────────────────────────────────
def build_response_sequences(
    df: pd.DataFrame,
    id_col: str = "NACCID",
    coding_key: str = "standard",
    codings: dict = CODINGS,
    save_csv: str | None = None,
) -> pd.DataFrame:
    rules = codings[coding_key]["rules"]

    sequences = (
        df.groupby(id_col)
        .apply(lambda grp: [label_visit(row, rules) for _, row in grp.iterrows()])
        .reset_index()
    )
    sequences.columns = [id_col, "Responses"]
    sequences["Coding"] = coding_key

    # Add occurrence count and sort by it descending for readable CSV output
    counts = sequences["Responses"].apply(tuple).map(
        sequences["Responses"].apply(tuple).value_counts()
    )
    sequences["Count"] = counts
    sequences = sequences.sort_values("Count", ascending=False).reset_index(drop=True)

    if save_csv:
        sequences.to_csv(save_csv, index=False)
        print(f"Saved → {save_csv}")

    return sequences

def summarise_sequences(
    sequences_df: pd.DataFrame,
    save_csv: str | None = None,
) -> pd.DataFrame:
    df = sequences_df.copy()
    # Convert lists to tuples so they can be used as groupby keys
    df["Responses"] = df["Responses"].apply(
        lambda x: tuple(x) if isinstance(x, list) else x
    )

    summary = (
        df
        .groupby(["Responses", "Coding"], as_index=False)["Count"]
        .first()
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    total = summary["Count"].sum()
    summary["Pct"] = (summary["Count"] / total * 100).round(2)

    if save_csv:
        summary.to_csv(save_csv, index=False)
        print(f"Saved → {save_csv}  ({len(summary)} unique patterns)")

    return summary


            
def convert_to_real_nan(row):
    try:
        if not isinstance(row, str): return row
        # Ensure it's a valid list format
        row = row.replace("nan", "null")  # Replace 'nan' string with 'null' for JSON compatibility
        values = json.loads(row)  # Parse JSON-formatted string into a list

        # Convert 'None' (from JSON) into np.nan for numerical operations
        return [np.nan if x is None else float(x) for x in values]
    
    except (ValueError, TypeError, json.JSONDecodeError):
        return row  # If it can't be converted, return the original
    
def impute_mmse(df, covariates):
    """Original convenience wrapper: fits and transforms in one call (use only pre-split or for exploration)."""
    mmse_imputer, df_out = fit_mmse_imputer(df, covariates)
    return df_out

def _pad_to_max(lists):
    """Pad a sequence of lists to uniform length with NaN, return (2-D array, original lengths)."""
    lengths = [len(x) if isinstance(x, list) else 1 for x in lists]
    max_len = max(lengths)
    padded = []
    for lst, n in zip(lists, lengths):
        row = list(lst) if isinstance(lst, list) else [lst]
        row += [np.nan] * (max_len - n)
        padded.append(row)
    return np.array(padded, dtype=float), lengths


def _build_mmse_matrix(df, covariates):
    """Helper: build the combined MMSE + covariate matrix.

    Handles variable-length list columns by padding to the longest
    sequence with NaN.  Returns (mmse_matrix, combined_data, n_mmse_cols, orig_lengths).
    """
    mmse_lists = df['NACCMMSE'].apply(convert_to_real_nan)
    mmse_matrix, orig_lengths = _pad_to_max(mmse_lists)
    covariate_matrix = []
    for col in covariates:
        if df[col].dtype == 'object' and str(df[col].iloc[0]).startswith('['):
            col_data = df[col].apply(convert_to_real_nan)
            col_matrix, _ = _pad_to_max(col_data)
            covariate_matrix.append(col_matrix)
        else:
            col_matrix = df[col].values.reshape(-1, 1)
            covariate_matrix.append(col_matrix)
    covariate_data = np.hstack([x if len(x.shape) > 1 else x.reshape(-1, 1) for x in covariate_matrix])
    combined_data = np.hstack([mmse_matrix, covariate_data])
    return mmse_matrix, combined_data, mmse_matrix.shape[1], orig_lengths

def _process_mmse_values(imputed_mmse, orig_lengths=None):
    """Post-process imputed MMSE: cap at 30.0, round to 2 decimals, trim to original lengths."""
    processed = []
    for i, row in enumerate(imputed_mmse):
        n = orig_lengths[i] if orig_lengths is not None else len(row)
        processed.append([min(round(float(val), 2), 30.0) for val in row[:n]])
    return processed

def create_target(progression, progression_type):
    """Return a binary target label (0 or 1) from a Progression tuple.

    progression_type 'AD' : 1 if the patient ever reached AD (any visit == 2)
    progression_type 'CN' : 1 if the patient ever progressed beyond CN (any visit > 0)
    """
    if isinstance(progression, str):
        progression = eval(progression)
    if progression_type == 'AD':
        return 1 if 2 in progression else 0
    return 1 if any(v > 0 for v in progression) else 0


def fit_mmse_imputer(df, covariates):
    """Fit an IterativeImputer on df's MMSE + covariates and return (fitted_imputer, imputed_df).
    
    Use this on the TRAINING set only. Then call transform_mmse() on the test set.
    """
    mmse_matrix, combined_data, n_mmse_cols, orig_lengths = _build_mmse_matrix(df, covariates)
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20,
        random_state=42,
        initial_strategy='mean'
    )
    imputed_data = imputer.fit_transform(combined_data)
    imputed_mmse = imputed_data[:, :n_mmse_cols]
    df = df.copy()
    df['NACCMMSE'] = _process_mmse_values(imputed_mmse, orig_lengths)
    return imputer, df

def transform_mmse(df, covariates, fitted_imputer):
    """Apply a previously fitted MMSE imputer to a new DataFrame (e.g. test set).
    
    The imputer must have been fit via fit_mmse_imputer() on the training set.
    """
    mmse_matrix, combined_data, n_mmse_cols, orig_lengths = _build_mmse_matrix(df, covariates)
    # The test set may have fewer visits than the training set (pooled datasets have
    # variable visit counts).  Pad extra columns with NaN so the fitted imputer —
    # which expects training-set feature width — does not raise a shape mismatch.
    n_expected = fitted_imputer.n_features_in_
    if combined_data.shape[1] < n_expected:
        pad = np.full((combined_data.shape[0], n_expected - combined_data.shape[1]), np.nan)
        combined_data = np.hstack([combined_data, pad])
    imputed_data = fitted_imputer.transform(combined_data)
    imputed_mmse = imputed_data[:, :n_mmse_cols]
    df = df.copy()
    df['NACCMMSE'] = _process_mmse_values(imputed_mmse, orig_lengths)
    return df

# Manually impute a single vector of categorical longitudinal variables
def m_impute(list):
    # check the need for imputation
    if any(np.isnan(x) for x in list) and any((not np.isnan(x)) for x in list):
        flag = 1
        # impute nan using closest neighbor (prioritize left neighbor/past point)
        while flag:
            flag=0
            copy=list.copy()
            for i, x in enumerate(list):
                # replace nan with any adjacent non-nan neighbor(prioritize left). If no such neighbor, flag=1 to repeat cycle.
                if np.isnan(x):
                    if i==0:
                        # no left neighbor case
                        if np.isnan(list[i+1]):
                            flag=1
                        else: copy[i]=list[i+1]
                    elif i==len(list)-1:
                        # no right neighbor case
                        if np.isnan(list[i-1]):
                            flag=1
                        else: copy[i]=list[i-1]
                    else:
                        # regular case
                        if not np.isnan(list[i-1]): copy[i]=list[i-1]
                        elif not np.isnan(list[i+1]): copy[i]=list[i+1]
                        else: flag=1    
            list=copy
    return list

# Manually impute GDS using mean
def impute_gds(list):
    # check the need for imputation
    if any(np.isnan(x) for x in list) and any((not np.isnan(x)) for x in list):
        sum=0
        count=0
        # Calculate average score to up to 2 decimal places
        for x in list:
            if (not np.isnan(x)): 
                sum+=x
                count+=1
        avg = round(float(sum) / count, 2)
        # replace nan with average score
        for i,x in enumerate(list):
            if (np.isnan(x)):
                list[i]=avg
    return list


# create new categorical longitudinal variables "hearing" and "vision"
# using HEARING HEARAID HEARWAID, VISION VISCORR VISWCORR.
def create_hv(df):

    # check if having required vars
    for v in ['HEARING', 'HEARAID', 'HEARWAID', 'VISION', 'VISCORR', 'VISWCORR']:
        if v not in df.columns :
            print("Cannot create hearing/vision: Missing HEARING HEARAID HEARWAID, VISION VISCORR VISWCORR")
            return df
    
    hearing=list()
    vision=list()
    for a, row in df.iterrows():
        for var in ['HEARING', 'HEARAID', 'HEARWAID', 'VISION', 'VISCORR', 'VISWCORR']:
            if isinstance(row[var], str):
                # Convert the string representation of the list to an actual list
                row[var] = eval(row[var])
        
        # timepoint vectors for each sample
        h=list()
        v=list()

        for i in range(len(row['HEARING'])):
            # classify hearing
            if np.isnan(row['HEARING'][i]):
                h.append(np.nan)
            elif row['HEARING'][i]==1:
                # normal hearing
                h.append(0)
            else:
                # abnormal hearing without aid, check if aid helps
                if np.isnan(row['HEARAID'][i]):
                    # unknown aid presence
                    if np.isnan(row['HEARWAID'][i]):
                        h.append(2)
                    elif row['HEARWAID'][i]==0:
                        h.append(2)
                    else: 
                        h.append(1)
                elif row['HEARAID'][i]==0:
                    # no aid
                    h.append(2)
                else: 
                    # has aid, check if helps
                    if np.isnan(row['HEARWAID'][i]):
                        h.append(2)
                    elif row['HEARWAID'][i]==0:
                        h.append(2)
                    else: 
                        h.append(1)

            # classify vision
            if np.isnan(row['VISION'][i]):
                v.append(np.nan)
            elif row['VISION'][i]==1:
                # normal vision
                v.append(0)
            else:
                # abnormal vision without aid, check if aid helps
                if np.isnan(row['VISCORR'][i]):
                    # unknown aid presence
                    if np.isnan(row['VISWCORR'][i]):
                        v.append(2)
                    elif row['VISWCORR'][i]==0:
                        v.append(2)
                    else: 
                        v.append(1)
                elif row['VISCORR'][i]==0:
                    # no aid
                    v.append(2)
                else: 
                    # has aid, check if helps
                    if np.isnan(row['VISWCORR'][i]):
                        v.append(2)
                    elif row['VISWCORR'][i]==0:
                        v.append(2)
                    else: 
                        v.append(1)
                
        hearing.append(h)
        vision.append(v)

    # expand df with new vars
    df['hearing']=hearing
    df['vision']=vision
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  CALLABLE PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# ── Sentinel-value cleaning helpers ────────────────────────────────────────────


# ── Subject-level aggregation ──────────────────────────────────────────────────

# Longitudinal columns stored as per-visit lists
_LONG_COLS = [
    'NACCBMI', 'NACCMMSE', 'NACCGDS', 'CDRSUM', 'TOBAC30',
    'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
    'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL',
    'HEARING', 'HEARAID', 'HEARWAID', 'VISION', 'VISCORR', 'VISWCORR',
    'NACCLIVS', 'COMMUN', 'ALCOHOL'
]

# Scalar columns — take the most-recent visit value
_SCALAR_COLS = [
    'SEX', 'EDUC', 'NACCFAM',
    'CVHATT', 'CVAFIB', 'DIABETES', 'HYPERCHO', 'HYPERTEN',
    'B12DEF', 'DEPD', 'ANX', 'NACCTBI', 'SMOKYRS', 'RACE', 'HISPANIC',
    'NACCNE4S',
]

# Numeric coding for label_visit string output → integer used in Progression
_LABEL_INT = {'CN': 0, 'MCI': 1, 'AD': 2, 'Unknown': 3}


def build_subject_df(
    source_csv: str,
    min_visits: int = 2,
    max_visits: int = None,
    coding_key: str = 'standard',
) -> pd.DataFrame:
    """
    Read *source_csv* (the raw NACC investigator file) and aggregate to one
    row per subject, matching the schema of the existing pooled CSVs plus
    the new columns months_since_baseline, NACCLIVS, and COMMUN.

    Parameters
    ----------
    source_csv  : path to investigator_ftldlbd_nacc72.csv (or equivalent)
    min_visits  : keep subjects with at least this many visits
    max_visits  : keep subjects with at most this many visits (None = no cap)
    coding_key  : which CODINGS scheme to use for Progression labels

    Returns
    -------
    pd.DataFrame  — one row per subject, columns:
        ID, Prog_ID, Progression, n_visits,
        age, SEX, EDUC, ALCOHOL, NACCFAM, <comorbidities>, SMOKYRS, RACE, HISPANIC,
        NACCBMI, NACCMMSE, NACCGDS, CDRSUM, TOBAC30, <FAQ cols>,
        HEARING, HEARAID, HEARWAID, VISION, VISCORR, VISWCORR,
        NACCLIVS, COMMUN, months_since_baseline
    """
    raw = pd.read_csv(source_csv, low_memory=False)

    # Sort visits chronologically within each subject
    raw = raw.sort_values(['NACCID', 'NACCVNUM'])

    rules = CODINGS[coding_key]['rules']
    long_present = [c for c in _LONG_COLS if c in raw.columns]
    scalar_present = [c for c in _SCALAR_COLS if c in raw.columns]

    rows = []
    for naccid, grp in raw.groupby('NACCID', sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)

        if min_visits is not None and n < min_visits:
            continue
        if max_visits is not None and n > max_visits:
            continue

        # Progression tuple (integer-coded)
        prog_labels = [_LABEL_INT.get(label_visit(r, rules), 3) for _, r in grp.iterrows()]
        progression = tuple(prog_labels)

        # Prog_ID: 1 if any visit is more advanced than the starting label
        start = prog_labels[0]
        prog_id = 1 if any(v > start for v in prog_labels) else 0

        # age at baseline (NACCAGEB is defined as age at initial visit)
        age = grp['NACCAGEB'].iloc[0] if 'NACCAGEB' in grp.columns else np.nan

        # months_since_baseline from NACCFDYS
        if 'NACCFDYS' in grp.columns:
            months_since_baseline = [round(float(d) / 30.44, 2) for d in grp['NACCFDYS']]
        else:
            months_since_baseline = [np.nan] * n

        # Longitudinal columns — list of per-visit values
        long_vals = {col: grp[col].tolist() for col in long_present}

        # Scalar columns — most recent non-null value
        scalar_vals = {}
        for col in scalar_present:
            series = grp[col].dropna()
            scalar_vals[col] = series.iloc[-1] if len(series) > 0 else np.nan

        row = {
            'ID': naccid,
            'Prog_ID': prog_id,
            'Progression': progression,
            'n_visits': n,
            'age': age,
            **scalar_vals,
            **long_vals,
            'months_since_baseline': months_since_baseline,
        }
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ── Dataset summary / query helper ────────────────────────────────────────────

def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table of patient counts broken down by n_visits and class.

    Classes:
        CN_nonprog  — Progression contains 0, Prog_ID == 0
        CN_prog     — Progression contains 0, Prog_ID == 1
        MCI_nonprog — Progression does not contain 0, Prog_ID == 0
        MCI_prog    — Progression does not contain 0, Prog_ID == 1

    Parameters
    ----------
    df : DataFrame produced by build_subject_df() or one of the pooled CSVs.
         Must have columns: Progression, Prog_ID, n_visits (or visit count
         derivable from Progression length).

    Returns
    -------
    pd.DataFrame with columns:
        n_visits, CN_nonprog, CN_prog, MCI_nonprog, MCI_prog, Total
    """
    work = df.copy()

    # Parse Progression if stored as string
    work['Progression'] = work['Progression'].apply(
        lambda p: eval(p) if isinstance(p, str) else p
    )

    # Derive n_visits from Progression length if column absent
    if 'n_visits' not in work.columns:
        work['n_visits'] = work['Progression'].apply(len)

    def _classify(row):
        prog = row['Progression']
        pid = row['Prog_ID']
        cn_starting = prog[0] == 0
        if cn_starting:
            return 'CN_nonprog' if pid == 0 else 'CN_prog'
        else:
            return 'MCI_nonprog' if pid == 0 else 'MCI_prog'

    work['_class'] = work.apply(_classify, axis=1)

    summary = (
        work.groupby(['n_visits', '_class'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=['CN_nonprog', 'CN_prog', 'MCI_nonprog', 'MCI_prog'], fill_value=0)
    )
    summary['Total'] = summary.sum(axis=1)
    return summary.reset_index().sort_values('n_visits').reset_index(drop=True)

def _parse_list_col(row):
    """Parse a string-encoded list into a real Python list, converting 'nan' → np.nan."""
    if not isinstance(row, str):
        return row
    try:
        row = row.replace("nan", "null")
        values = json.loads(row)
        return [np.nan if x is None else float(x) for x in values]
    except (ValueError, TypeError, json.JSONDecodeError):
        return row

def _clean_sentinel(row, bad_vals):
    """Replace sentinel values in a list with np.nan."""
    try:
        if isinstance(row, str):
            row = ast.literal_eval(row)
        return [np.nan if val in bad_vals else val for val in row] if isinstance(row, list) else row
    except (ValueError, SyntaxError):
        return row

def _impute_bmi(values):
    """Impute NaN entries in a BMI list with the mean of valid entries."""
    if isinstance(values, list):
        valid = [x for x in values if not pd.isna(x)]
        if valid:
            avg = round(sum(valid) / len(valid), 1)
            return [round(avg, 1) if pd.isna(x) else round(x, 1) for x in values]
    return values


def _get_progression(row):
    """Safely parse the Progression column."""
    prog = row['Progression']
    if isinstance(prog, str):
        prog = eval(prog)
    return prog

def _cn_starting_indices(df):
    """Return iloc indices of rows whose first visit label is CN (0)."""
    idx = []
    for i, (_, r) in enumerate(df.iterrows()):
        prog = _get_progression(r)
        if prog[0] == 0:
            idx.append(i)
    return idx

def _detect_reverters(df, task):
    
    """Return iloc indices of reverters.
    task='CN_MCI': progressor (Prog_ID=1) ending at 0.
    task='MCI_AD': progressor (Prog_ID=1) ending at 1.
    """
    if task == 'CN_MCI':
        end_label = 0
        idx = []
        for i, (_, r) in enumerate(df.iterrows()):
            prog = _get_progression(r)
            pid = r['Prog_ID']
            if isinstance(pid, str):
                pid = eval(pid)
            if pid == 1 and prog[-1] == end_label:
                idx.append(i)
        return idx
    elif task == 'MCI_AD':
        idx = []
        for i, (_, r) in enumerate(df.iterrows()):
            prog = _get_progression(r)
            pid = r['Prog_ID']
            if isinstance(pid, str):
                pid = eval(pid)
            if pid == 1 and (prog[-1] == 1 or prog[-1] == 0):
                idx.append(i)
        return idx
    


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    source_csv: str,
    dest_dir: str = "Dataset_output",
    min_visits: int = 2,
    max_visits: int = None,
    min_age: int = 50,
    lead_time_pct: float = 0.05,
    do_impute: bool = False,
    verbose: bool = True,
):
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    source_csv      : path to the raw NACC investigator CSV
                      (e.g. 'investigator_ftldlbd_nacc72.csv').
    dest_dir        : output folder (created if needed).
    min_visits      : minimum visit count to include.
    max_visits      : maximum visit count to include (None = no cap).
    min_age         : drop subjects younger than this at baseline.
    lead_time_pct   : fraction of each (visit_count × class) cell to reserve
                      for the lead-time holdout set (default 5%).
    do_impute       : if True, apply GDS/categorical/BMI imputation.
    verbose         : print progress.

    Outputs (written to *dest_dir*)
    -------
    pooled_CN.csv          — CN-starting patients (Prog_ID 0/1).
    pooled_MCI_AD.csv      — MCI-starting patients.
    lead_time_CN.csv       — 5% lead-time holdout, CN-starting.
    lead_time_MCI_AD.csv   — 5% lead-time holdout, MCI-starting.
    reverters_CN.csv, reverters_MCI_AD.csv
    """
    import math

    os.makedirs(dest_dir, exist_ok=True)
    stats = {"age_dropped": {}}
    reverters_cn = []
    reverters_mci_ad = []
    pool_cn = []
    pool_mci_ad = []
    lead_cn = []
    lead_mci_ad = []

    FAQ_COLS = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
                'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
    HV_RAW   = ['HEARING', 'HEARAID', 'HEARWAID', 'VISION', 'VISCORR', 'VISWCORR']

    # ── Build one-row-per-subject DataFrame from source CSV ───────────────
    if verbose:
        print(f"Building subject DataFrame from {source_csv} ...")
    subject_df = build_subject_df(
        source_csv,
        min_visits=min_visits,
        max_visits=max_visits,
    )
    if verbose:
        print(f"  {len(subject_df)} subjects ({subject_df['n_visits'].min()}–"
              f"{subject_df['n_visits'].max()} visits)")

    # ── Clean scalar sentinel values ──────────────────────────────────────
    # SMOKYRS: 888 = "not assessed", -4 = "not available" → NaN
    if 'SMOKYRS' in subject_df.columns:
        subject_df['SMOKYRS'] = subject_df['SMOKYRS'].replace([-4, 888], np.nan)

    # NACCNE4S: 9 = missing/unknown → NaN; cast to nullable Int64 so values
    # serialize as integers (0, 1, 2) rather than floats (0.0, 1.0, 2.0)
    if 'NACCNE4S' in subject_df.columns:
        subject_df['NACCNE4S'] = subject_df['NACCNE4S'].replace(9, np.nan).astype('Int64')

    # Comorbidity columns — binary recode:
    #   1 (Recent/Active)   → 1
    #   0 (Absent), 2 (Remote/Inactive), 8, 9 (Unknown), -4 (N/A) → 0
    # Only an active/recent condition is treated as present.
    _COMORBIDITY_COLS = [
        'NACCFAM', 'CVHATT', 'CVAFIB', 'DIABETES',
        'HYPERCHO', 'HYPERTEN', 'B12DEF', 'DEPD', 'ANX', 'NACCTBI',
    ]
    for col in _COMORBIDITY_COLS:
        if col in subject_df.columns:
            subject_df[col] = (subject_df[col] == 1).astype(int)

    # ── Remove Unknown-labelled subjects ──────────────────────────────────
    # Label 3 = Unknown (no coding rule matched). These subjects have at least
    # one visit that is neither CN (0), MCI (1), nor AD (2) and must be excluded.
    before = len(subject_df)
    subject_df = subject_df[subject_df['Progression'].apply(
        lambda p: 3 not in (eval(p) if isinstance(p, str) else p)
    )].copy()
    if verbose:
        removed = before - len(subject_df)
        if removed:
            print(f"  Removed {removed} subjects with Unknown-labelled visits")

    # ── Process each visit-count group ────────────────────────────────────
    for n_visits, df in subject_df.groupby('n_visits'):
        df = df.copy()
        label = f"{n_visits}visit"
        if verbose:
            print(f"── {label} ({len(df)} subjects) ", end="")

        # ── 1. Parse list-string columns & clean sentinels ───────────────
        # build_subject_df stores raw numeric values; sentinel codes must be
        # converted to NaN before any NaN-based filters or imputation below.
        for col in _LONG_COLS:
            if col in df.columns:
                df[col] = df[col].apply(_parse_list_col)

        # Column-specific sentinel sets (NACC data dictionary)
        df['NACCBMI']  = df['NACCBMI'].apply(lambda r: _clean_sentinel(r, {-4, 888, 888.8}))
        df['NACCMMSE'] = df['NACCMMSE'].apply(lambda r: _clean_sentinel(r, {-4, 88, 95, 96, 97, 98, 888}))
        df['NACCGDS']  = df['NACCGDS'].apply(lambda r: _clean_sentinel(r, {-4, 88}))
        df['CDRSUM']   = df['CDRSUM'].apply(lambda r: _clean_sentinel(r, {-4}))
        df['TOBAC30']  = df['TOBAC30'].apply(lambda r: _clean_sentinel(r, {-4, 9}))
        df['NACCLIVS'] = df['NACCLIVS'].apply(lambda r: _clean_sentinel(r, {-4, 8, 9}))
        df['ALCOHOL'] = df['ALCOHOL'].apply(lambda r: _clean_sentinel(r, {-4, 9}))
        df['COMMUN'] = df['COMMUN'].apply(lambda r: _clean_sentinel(r, {-4, 9}))
        for col in FAQ_COLS:
            if col in df.columns:
                df[col] = df[col].apply(lambda r: _clean_sentinel(r, {-4, 9, 8}))
        for col in HV_RAW:
            if col in df.columns:
                df[col] = df[col].apply(lambda r: _clean_sentinel(r, {-4, 9, 8}))

        # ── 2. Drop subjects below min_age ────────────────────────────────
        n0 = len(df)
        df = df[df['age'] >= min_age]
        stats["age_dropped"][label] = n0 - len(df)

        if df.empty:
            if verbose:
                print("→ empty after filtering, skipped")
            continue

        # ── 3. Optional imputation ────────────────────────────────────────
        if do_impute:
            imp_cats = ['TOBAC30'] + FAQ_COLS
            df['NACCGDS'] = df['NACCGDS'].apply(
                lambda x: eval(x.replace("nan", "np.nan")) if isinstance(x, str) else x
            ).apply(impute_gds)
            for v in imp_cats:
                if v in df.columns:
                    df[v] = df[v].apply(
                        lambda x: eval(x.replace("nan", "np.nan")) if isinstance(x, str) else x
                    ).apply(m_impute)
            df['NACCBMI'] = df['NACCBMI'].apply(_impute_bmi)

        # ── 4. Create hearing / vision composites ─────────────────────────
        df = create_hv(df)
        df.drop(columns=HV_RAW, inplace=True, errors='ignore')

        if verbose:
            print(f"→ {len(df)} rows")

        # ── 5. Split CN-starting vs MCI-starting ─────────────────────────
        # CN group: first visit label == 0
        # MCI group: first visit label == 1 (AD-only starters discarded)
        cn_idx = _cn_starting_indices(df)
        cn_group = df.iloc[cn_idx].copy()
        rest = df.drop(df.index[cn_idx])
        mci_idx = [i for i, (_, r) in enumerate(rest.iterrows())
                   if _get_progression(r)[0] == 1]
        mci_group = rest.iloc[mci_idx].copy()

        # ── 6. Remove reverters ───────────────────────────────────────────
        rev_cn = _detect_reverters(cn_group, 'CN_MCI')
        rev_ma = _detect_reverters(mci_group, 'MCI_AD')
        if rev_cn:
            reverters_cn.append(cn_group.iloc[rev_cn])
            cn_group = cn_group.drop(cn_group.index[rev_cn])
        if rev_ma:
            reverters_mci_ad.append(mci_group.iloc[rev_ma])
            mci_group = mci_group.drop(mci_group.index[rev_ma])

        # ── 7. 5% lead-time stratified split ─────────────────────────────
        # Four classes within each group; sample ceil(5%) from each class.
        def _lead_split(group):
            """Return (lead_df, pooled_df) with stratified 5% lead-time sample."""
            lead_parts, pool_parts = [], []
            for pid in [0, 1]:
                subset = group[group['Prog_ID'] == pid]
                if subset.empty:
                    continue
                k = math.ceil(len(subset) * lead_time_pct)
                sampled = subset.sample(n=k, random_state=42)
                lead_parts.append(sampled)
                pool_parts.append(subset.drop(sampled.index))
            lead_df = pd.concat(lead_parts) if lead_parts else pd.DataFrame()
            pool_df = pd.concat(pool_parts) if pool_parts else pd.DataFrame()
            return lead_df, pool_df

        cn_lead, cn_pool   = _lead_split(cn_group)
        mci_lead, mci_pool = _lead_split(mci_group)

        if not cn_pool.empty:   pool_cn.append(cn_pool)
        if not mci_pool.empty:  pool_mci_ad.append(mci_pool)
        if not cn_lead.empty:   lead_cn.append(cn_lead)
        if not mci_lead.empty:  lead_mci_ad.append(mci_lead)

    # ── Save pooled datasets ──────────────────────────────────────────────
    def _save(frames, fname, label):
        if not frames:
            return
        out = pd.concat(frames, axis=0).reset_index(drop=True)
        assert out['ID'].is_unique, \
            f"Duplicate IDs in {fname}: {out['ID'][out['ID'].duplicated()].tolist()}"
        out.to_csv(os.path.join(dest_dir, fname), index=False)
        if verbose:
            n_prog = int((out['Prog_ID'] == 1).sum())
            print(f"  {fname}: {len(out)} rows (prog: {n_prog}, non-prog: {len(out) - n_prog})")

    _save(pool_cn,    "pooled_CN.csv",        "pooled CN")
    _save(pool_mci_ad, "pooled_MCI_AD.csv",   "pooled MCI/AD")
    _save(lead_cn,    "lead_time_CN.csv",     "lead-time CN")
    _save(lead_mci_ad, "lead_time_MCI_AD.csv", "lead-time MCI/AD")

    # ── Save reverters ────────────────────────────────────────────────────
    if reverters_cn:
        pd.concat(reverters_cn).to_csv(os.path.join(dest_dir, "reverters_CN.csv"), index=False)
    if reverters_mci_ad:
        pd.concat(reverters_mci_ad).to_csv(os.path.join(dest_dir, "reverters_MCI_AD.csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print("\n── Drop summary ──")
        for key in ("age_dropped",):
            for f, n in stats[key].items():
                if n:
                    print(f"  {f}: {n} rows ({key.replace('_', ' ')})")        
        n_rev_cn = sum(len(r) for r in reverters_cn) if reverters_cn else 0
        n_rev_ma = sum(len(r) for r in reverters_mci_ad) if reverters_mci_ad else 0
        print(f"  Reverters saved — CN: {n_rev_cn}, MCI/AD: {n_rev_ma}")
        print(f"\n✓ Output written to {dest_dir}/")

    return stats