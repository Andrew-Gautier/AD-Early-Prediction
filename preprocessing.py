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

def combine_ids_to_one_row(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, header=0)
            
            # Combine all IDs into a single row
            all_ids = []
            for ids in df['NACCID']:
                ids_list = eval(ids) if isinstance(ids, str) else ids
                all_ids.extend(ids_list)
            
            # Create a new DataFrame with a single row of combined IDs
            combined_df = pd.DataFrame({'NACCID': [str(all_ids)]})
            
            # Save the updated DataFrame back to the CSV file
            combined_df.to_csv(file_path, index=False)
            
def clean_and_parse_csv(file_path):
    ids = []
    vectors = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header
        for line in file:
            if line.startswith('"['):
                # Extract and clean the NACCIDs
                id_str = line.split('","')[0].strip().strip('"').strip('[]').replace("'", "")
                id_list = id_str.split(", ")
                ids.extend(id_list)
                
                # Extract and clean the ProgressionVector
                vector_str = line.split('","')[1].strip().strip('"').strip('[]')
                vector = ast.literal_eval(vector_str)
                vectors.extend([vector] * len(id_list))
    
    # Create a DataFrame with cleaned data
    df = pd.DataFrame({'NACCID': ids, 'ProgressionVector': vectors})
    return df

def clean_all_csv_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df_cleaned = clean_and_parse_csv(file_path)
            cleaned_file_path = os.path.join(directory, f"cleaned_{filename}")
            df_cleaned.to_csv(cleaned_file_path, index=False)
            print(f"Cleaned file saved to: {cleaned_file_path}")
            
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
    mmse_lists = df['MMSE'].apply(convert_to_real_nan)
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
    df['MMSE'] = _process_mmse_values(imputed_mmse, orig_lengths)
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
    df['MMSE'] = _process_mmse_values(imputed_mmse, orig_lengths)
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

# Slice eligible samples from greater-visits dataset to select number of continuous time points
# target_class: 0 for non-progressor in MCI/AD, 
#               1 for progressor in CN/MCI
def time_series_slicer(target_class, original_data, target_point_count):

    # filter out ineligible samples
    if target_class:
        eligible_ind = []
        for i, (ind ,r) in enumerate(original_data.iterrows()):
            # keep progressors with label 0 and 1
            progression = r['Progression']
            if isinstance(progression, str):
                # Convert the string representation of the list to an actual list
                progression = eval(progression)
            if (0 in progression) and (1 in progression): eligible_ind.append(i)
        sliced_data = original_data.iloc[eligible_ind]
    else:
        eligible_ind = []
        for i,(ind,r) in enumerate(original_data.iterrows()):
            # keep non-progressor with only label 1
            progression = r['Progression']
            if isinstance(progression, str):
                # Convert the string representation of the list to an actual list
                progression = eval(progression)
            if (not 0 in progression) and (not 2 in progression): eligible_ind.append(i)
        sliced_data = original_data.iloc[eligible_ind]
    
    # grand dataset needed for age adjustment
    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    ### NOTE: MAGIC FILE PATH BELOW, CHANGE IF NECESSARY
    target_file = os.path.join(current_dir, 'investigator_ftldlbd_nacc72.csv')
    df = pd.read_csv(target_file, header=0)

    for ind, (i, row) in enumerate(sliced_data.iterrows()):
        if isinstance(row['Progression'], str):
            # Convert the string representation of the list to an actual list
            row['Progression'] = eval(row['Progression'])
        old_tp=0
        new_tp=0
        x = row['Progression']
        if target_class:
            # for progressors in CN/MCI
            if any(1==x[a] for a in range(target_point_count)):
                # if in first n visits, use the first n time points
                old_tp=0
                new_tp=target_point_count-1
            else:
                # elsewise, use the first progressed visit and n-1 visits ahead of it
                for a in range(target_point_count, len(x)):
                    if x[a]==1:
                        new_tp=a
                        old_tp= a - target_point_count +1
                        break
        else:
            # for non-progressors in MCI/AD, use the most distant n visits
            old_tp = 0
            new_tp = target_point_count-1

        
        # slice time points for all longitudinal variables
        tags = ['Progression', 'BMI', 'MMSE', 'GDS', 'CDR', "TOBAC30", "BILLS", 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL','hearing','vision']
        for v in tags:
            if isinstance(row[v], str):
                row[v]=eval(row[v].replace("nan", "np.nan"))
            row[v]=row[v][old_tp:new_tp+1]
        
        # age adjustment using NACCAGE
        row['age']=df[df['NACCID']==row['ID']].sort_values(by="NACCAGE", ascending=True).iloc[old_tp]['NACCAGE']

        sliced_data.iloc[ind] = row
    return sliced_data

# create new categorical longitudinal variables "hearing" and "vision"
#      using HEARING HEARAID HEARWAID, VISION VISCORR VISWCORR.
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

def _split_mci_ad_np(df):
    """Split out MCI/AD non-progressors (only label 1 throughout) for special MMSE rules."""
    np_idx = []
    for i, (_, r) in enumerate(df.iterrows()):
        prog = r['Progression']
        if isinstance(prog, str):
            prog = eval(prog)
        if 0 not in prog and 2 not in prog:
            np_idx.append(i)
    df_np = df.iloc[np_idx]
    df_rest = df.drop(df.index[np_idx])
    return df_rest, df_np

def _get_progression(row):
    """Safely parse the Progression column."""
    prog = row['Progression']
    if isinstance(prog, str):
        prog = eval(prog)
    return prog

def _cn_starting_indices(df):
    """Return iloc indices of rows where label 0 (CN) appears in Progression.
    This captures all CN-starting patients: stable CN, CN→MCI, CN→AD, CN→MCI→AD."""
    idx = []
    for i, (_, r) in enumerate(df.iterrows()):
        prog = _get_progression(r)
        if 0 in prog:
            idx.append(i)
    return idx

def _detect_reverters(df, task):
    """Return iloc indices of reverters.
    task='CN_MCI': progressor (Prog_ID=1) ending at 0.
    task='MCI_AD': progressor (Prog_ID=1) ending at 1.
    """
    end_label = 0 if task == 'CN_MCI' else 1
    idx = []
    for i, (_, r) in enumerate(df.iterrows()):
        prog = _get_progression(r)
        pid = r['Prog_ID']
        if isinstance(pid, str):
            pid = eval(pid)
        if pid == 1 and prog[-1] == end_label:
            idx.append(i)
    return idx


# ── Lead-time slicing helpers ──────────────────────────────────────────────────

_LT_TAGS = ['Progression', 'BMI', 'MMSE', 'GDS', 'CDR', 'TOBAC30', 'BILLS',
            'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS',
            'PAYATTN', 'REMDATES', 'TRAVEL', 'hearing', 'vision']


def _slice_first_n(df, n):
    """Slice all longitudinal columns in *df* to the first *n* time points (in-place)."""
    for ind, (i, row) in enumerate(df.iterrows()):
        for v in _LT_TAGS:
            val = row[v]
            if isinstance(val, str):
                val = eval(val.replace("nan", "np.nan"))
            row[v] = val[:n]
        df.iloc[ind] = row


def _slice_progressor(df, n, target_label=2):
    """Slice progressors to *n* visits around the first occurrence of *target_label* (in-place).

    If the target label appears within the first *n* visits, use visits 0..n-1.
    Otherwise, end at the first target visit and go back n-1 visits.
    Mirrors the logic in ``time_series_slicer`` for target_class=1.
    """
    for ind, (i, row) in enumerate(df.iterrows()):
        prog = row['Progression']
        if isinstance(prog, str):
            prog = eval(prog)
        # find window
        if any(prog[a] == target_label for a in range(min(n, len(prog)))):
            old_tp, new_tp = 0, n - 1
        else:
            old_tp, new_tp = 0, n - 1  # fallback
            for a in range(n, len(prog)):
                if prog[a] == target_label:
                    new_tp = a
                    old_tp = a - n + 1
                    break
        for v in _LT_TAGS:
            val = row[v]
            if isinstance(val, str):
                val = eval(val.replace("nan", "np.nan"))
            row[v] = val[old_tp:new_tp + 1]
        df.iloc[ind] = row


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(
    source_dir: str = "Raw_Cohorts",
    dest_dir: str = "Dataset_output",
    min_visits: int = 2,
    max_visits: int = 10,
    min_age: int = 50,
    do_impute: bool = False,
    verbose: bool = True,
):
    """
    End-to-end preprocessing pipeline.

    Parameters
    ----------
    source_dir : path to folder with ``{N}visit_combined.csv`` files.
    dest_dir   : output folder (created if needed).
    min_visits / max_visits : range of visit-count files to process (inclusive).
    min_age    : drop subjects younger than this at baseline.
    do_impute  : if True, apply GDS/categorical/BMI imputation; else leave NaN.
    verbose    : print progress.

    Outputs (written to *dest_dir*)
    -------
    ``pooled_CN.csv``          — 2–5 visit CN-starting patients (Prog_ID distinguishes prog/non-prog).
    ``pooled_MCI_AD.csv``      — 2–5 visit MCI-starting patients.
    ``lead_time_CN.csv``       — 6–10 visit CN-starting patients (full sequences).
    ``lead_time_MCI_AD.csv``   — 6–10 visit MCI-starting patients (full sequences).
    ``reverters_CN.csv``, ``reverters_MCI_AD.csv`` (all reverters pooled).

    CN model inclusion
    ------------------
    CN-starting patients include anyone whose Progression contains label 0.
    Progressors (Prog_ID=1) are those who reach MCI (1) or AD (2) at any visit.
    Reverters (Prog_ID=1 but ending at 0) are excluded.

    MCI→AD model is unchanged: starts MCI (no 0 in Progression), progressors reach AD (2).
    """
    os.makedirs(dest_dir, exist_ok=True)
    stats = {"bmi_dropped": {}, "mmse_dropped": {}, "age_dropped": {}}
    reverters_cn = []
    reverters_mci_ad = []
    pool_cn = []       # accumulates CN-starting DataFrames from 2–5 visit files
    pool_mci_ad = []   # accumulates MCI-starting DataFrames from 2–5 visit files
    lead_cn = []       # accumulates CN-starting DataFrames from 6–10 visit files (full sequences)
    lead_mci_ad = []   # accumulates MCI-starting DataFrames from 6–10 visit files (full sequences)

    # --- MMSE tolerance map: visit_count → max allowed NaN -----------------
    mmse_tol = {2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4}

    # Longitudinal columns that get FAQ-style cleaning (-4, 9, 8 → NaN)
    FAQ_COLS = ['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
                'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
    HV_RAW   = ['HEARING', 'HEARAID', 'HEARWAID', 'VISION', 'VISCORR', 'VISWCORR']

    files = [f"{n}visit_combined.csv" for n in range(min_visits, max_visits + 1)]

    for fname in files:
        fpath = os.path.join(source_dir, fname)
        if not os.path.exists(fpath):
            if verbose:
                print(f"  ⚠ {fname} not found, skipping")
            continue
        n_visits = int(fname.split("visit")[0])
        if verbose:
            print(f"── {fname} ", end="")

        df = pd.read_csv(fpath)

        # Drop unnamed index column if present
        if df.columns[0].startswith("Unnamed"):
            df.drop(df.columns[0], axis=1, inplace=True)

        # ── 1. Parse list-string columns ──────────────────────────────────
        df['BMI']  = df['BMI'].apply(_parse_list_col)
        df['MMSE'] = df['MMSE'].apply(_parse_list_col)

        # ── 2. Drop rows where ALL BMI entries are NaN ────────────────────
        n0 = len(df)
        df = df[df['BMI'].apply(lambda x: not all(pd.isna(v) for v in x) if isinstance(x, list) else True)]
        stats["bmi_dropped"][fname] = n0 - len(df)

        # ── 3. Drop subjects below min_age ────────────────────────────────
        n0 = len(df)
        df = df[df['age'] >= min_age]
        stats["age_dropped"][fname] = n0 - len(df)

        # ── 4. MMSE tolerance filtering ───────────────────────────────────
        tol = mmse_tol.get(n_visits, 4)
        n0 = len(df)
        if n_visits >= 6:
            # split MCI/AD NP for special rule (allow 1 NaN in first 5 visits)
            df_main, df_np = _split_mci_ad_np(df)
            df_main = df_main[df_main['MMSE'].apply(
                lambda x: sum(pd.isna(v) for v in x) <= tol if isinstance(x, list) else True)]
            if not df_np.empty:
                df_np = df_np[df_np['MMSE'].apply(
                    lambda x: (sum(pd.isna(v) for v in x) <= tol or
                               sum(pd.isna(v) for v in x[:5]) <= 1)
                    if isinstance(x, list) else True)]
            df = pd.concat([df_main, df_np], axis=0)
        else:
            df = df[df['MMSE'].apply(
                lambda x: sum(pd.isna(v) for v in x) <= tol if isinstance(x, list) else True)]
        stats["mmse_dropped"][fname] = n0 - len(df)

        if df.empty:
            if verbose:
                print("→ empty after filtering, skipped")
            continue

        # ── 5. Sentinel-value cleaning ────────────────────────────────────
        df['GDS']    = df['GDS'].apply(lambda r: _clean_sentinel(r, {-4, 88}))
        df['TOBAC30'] = df['TOBAC30'].apply(lambda r: _clean_sentinel(r, {-4, 9}))
        for col in FAQ_COLS:
            df[col] = df[col].apply(lambda r: _clean_sentinel(r, {-4, 9, 8}))
        for col in HV_RAW:
            df[col] = df[col].apply(lambda r: _clean_sentinel(r, {-4, 9, 8}))

        # ── 6. Optional imputation ────────────────────────────────────────
        if do_impute:
            imp_cats = ['TOBAC30'] + FAQ_COLS + HV_RAW
            df['GDS'] = df['GDS'].apply(
                lambda x: eval(x.replace("nan", "np.nan")) if isinstance(x, str) else x
            ).apply(impute_gds)
            for v in imp_cats:
                df[v] = df[v].apply(
                    lambda x: eval(x.replace("nan", "np.nan")) if isinstance(x, str) else x
                ).apply(m_impute)
            df['BMI'] = df['BMI'].apply(_impute_bmi)

        # ── 7. Create hearing / vision composites ─────────────────────────
        create_hv(df)
        df.drop(columns=HV_RAW, inplace=True, errors='ignore')

        if verbose:
            print(f"→ {len(df)} rows")

        # ── 8. Split into CN-starting vs MCI-starting ────────────────────
        cn_idx = _cn_starting_indices(df)
        cn_group = df.iloc[cn_idx].copy()
        mci_group = df.drop(df.index[cn_idx]).copy()

        # detect and remove reverters
        rev_cn = _detect_reverters(cn_group, 'CN_MCI')
        rev_ma = _detect_reverters(mci_group, 'MCI_AD')
        if rev_cn:
            reverters_cn.append(cn_group.iloc[rev_cn])
            cn_group = cn_group.drop(cn_group.index[rev_cn])
        if rev_ma:
            reverters_mci_ad.append(mci_group.iloc[rev_ma])
            mci_group = mci_group.drop(mci_group.index[rev_ma])

        # ── 9. Route to pooled (2–5) or lead-time (6–10) accumulators ────
        if n_visits <= 5:
            if not cn_group.empty:
                pool_cn.append(cn_group)
            if not mci_group.empty:
                pool_mci_ad.append(mci_group)
        else:
            # 6–10 visits: keep full sequences for lead-time testing
            if not cn_group.empty:
                lead_cn.append(cn_group)
            if not mci_group.empty:
                lead_mci_ad.append(mci_group)

    # ── Save pooled 2–5 visit datasets ────────────────────────────────────
    if pool_cn:
        pooled_cn_df = pd.concat(pool_cn, axis=0).reset_index(drop=True)
        assert pooled_cn_df['ID'].is_unique, \
            f"Duplicate IDs in pooled_CN: {pooled_cn_df['ID'][pooled_cn_df['ID'].duplicated()].tolist()}"
        pooled_cn_df.to_csv(os.path.join(dest_dir, "pooled_CN.csv"), index=False)
        if verbose:
            n_prog = int((pooled_cn_df['Prog_ID'] == 1).sum())
            print(f"  pooled_CN.csv: {len(pooled_cn_df)} rows "
                  f"(prog: {n_prog}, non-prog: {len(pooled_cn_df) - n_prog})")

    if pool_mci_ad:
        pooled_mci_df = pd.concat(pool_mci_ad, axis=0).reset_index(drop=True)
        assert pooled_mci_df['ID'].is_unique, \
            f"Duplicate IDs in pooled_MCI_AD: {pooled_mci_df['ID'][pooled_mci_df['ID'].duplicated()].tolist()}"
        pooled_mci_df.to_csv(os.path.join(dest_dir, "pooled_MCI_AD.csv"), index=False)
        if verbose:
            n_prog = int((pooled_mci_df['Prog_ID'] == 1).sum())
            print(f"  pooled_MCI_AD.csv: {len(pooled_mci_df)} rows "
                  f"(prog: {n_prog}, non-prog: {len(pooled_mci_df) - n_prog})")

    # ── Save lead-time datasets (6–10 visits, full sequences) ─────────────
    if lead_cn:
        lead_cn_df = pd.concat(lead_cn, axis=0).reset_index(drop=True)
        lead_cn_df.to_csv(os.path.join(dest_dir, "lead_time_CN.csv"), index=False)
        if verbose:
            n_prog = int((lead_cn_df['Prog_ID'] == 1).sum())
            print(f"  lead_time_CN.csv: {len(lead_cn_df)} rows "
                  f"(prog: {n_prog}, non-prog: {len(lead_cn_df) - n_prog})")

    if lead_mci_ad:
        lead_mci_df = pd.concat(lead_mci_ad, axis=0).reset_index(drop=True)
        lead_mci_df.to_csv(os.path.join(dest_dir, "lead_time_MCI_AD.csv"), index=False)
        if verbose:
            n_prog = int((lead_mci_df['Prog_ID'] == 1).sum())
            print(f"  lead_time_MCI_AD.csv: {len(lead_mci_df)} rows "
                  f"(prog: {n_prog}, non-prog: {len(lead_mci_df) - n_prog})")

    # ── Save reverters ────────────────────────────────────────────────────
    if reverters_cn:
        pd.concat(reverters_cn).to_csv(os.path.join(dest_dir, "reverters_CN.csv"), index=False)
    if reverters_mci_ad:
        pd.concat(reverters_mci_ad).to_csv(os.path.join(dest_dir, "reverters_MCI_AD.csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print("\n── Drop summary ──")
        for key in ("bmi_dropped", "mmse_dropped", "age_dropped"):
            for f, n in stats[key].items():
                if n:
                    print(f"  {f}: {n} rows ({key.replace('_', ' ')})")
        n_rev_cn = sum(len(r) for r in reverters_cn) if reverters_cn else 0
        n_rev_ma = sum(len(r) for r in reverters_mci_ad) if reverters_mci_ad else 0
        print(f"  Reverters saved — CN: {n_rev_cn}, MCI/AD: {n_rev_ma}")
        print(f"\n✓ Output written to {dest_dir}/")

    return stats