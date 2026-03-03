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

def _build_mmse_matrix(df, covariates):
    """Helper: build the combined MMSE + covariate matrix and return (mmse_matrix, combined_data, n_mmse_cols)."""
    mmse_lists = df['MMSE'].apply(convert_to_real_nan)
    mmse_matrix = np.array([x for x in mmse_lists])
    covariate_matrix = []
    for col in covariates:
        if df[col].dtype == 'object' and str(df[col].iloc[0]).startswith('['):
            col_data = df[col].apply(convert_to_real_nan)
            col_matrix = np.array([x for x in col_data])
            covariate_matrix.append(col_matrix)
        else:
            col_matrix = df[col].values.reshape(-1, 1)
            covariate_matrix.append(col_matrix)
    covariate_data = np.hstack([x if len(x.shape) > 1 else x.reshape(-1, 1) for x in covariate_matrix])
    combined_data = np.hstack([mmse_matrix, covariate_data])
    return mmse_matrix, combined_data, mmse_matrix.shape[1]

def _process_mmse_values(imputed_mmse):
    """Post-process imputed MMSE: cap at 30.0, round to 2 decimals, convert to float lists."""
    processed = []
    for row in imputed_mmse:
        processed.append([min(round(float(val), 2), 30.0) for val in row])
    return processed

def fit_mmse_imputer(df, covariates):
    """Fit an IterativeImputer on df's MMSE + covariates and return (fitted_imputer, imputed_df).
    
    Use this on the TRAINING set only. Then call transform_mmse() on the test set.
    """
    mmse_matrix, combined_data, n_mmse_cols = _build_mmse_matrix(df, covariates)
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20,
        random_state=42,
        initial_strategy='mean'
    )
    imputed_data = imputer.fit_transform(combined_data)
    imputed_mmse = imputed_data[:, :n_mmse_cols]
    df = df.copy()
    df['MMSE'] = _process_mmse_values(imputed_mmse)
    return imputer, df

def transform_mmse(df, covariates, fitted_imputer):
    """Apply a previously fitted MMSE imputer to a new DataFrame (e.g. test set).
    
    The imputer must have been fit via fit_mmse_imputer() on the training set.
    """
    mmse_matrix, combined_data, n_mmse_cols = _build_mmse_matrix(df, covariates)
    imputed_data = fitted_imputer.transform(combined_data)
    imputed_mmse = imputed_data[:, :n_mmse_cols]
    df = df.copy()
    df['MMSE'] = _process_mmse_values(imputed_mmse)
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

def _cn_mci_indices(df):
    """Return iloc indices of rows containing label 0 (CN) in Progression → CN/MCI group."""
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
    ``{N}visit_CN_MCI.csv``, ``{N}visit_MCI_AD.csv`` for 2–4 visits.
    ``original_5visit_CN_MCI.csv``, ``original_5visit_MCI_AD.csv``.
    ``5visit_CN_MCI.csv``, ``5visit_MCI_AD.csv`` (augmented with 6–10 slices).
    ``reverters_CN_MCI.csv``, ``reverters_MCI_AD.csv`` (all reverters pooled).
    """
    os.makedirs(dest_dir, exist_ok=True)
    stats = {"bmi_dropped": {}, "mmse_dropped": {}, "age_dropped": {}}
    reverters_cn_mci = []
    reverters_mci_ad = []
    new_5_cn_mci = pd.DataFrame()
    new_5_mci_ad = pd.DataFrame()

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

        # ── 8. Split & save ───────────────────────────────────────────────
        if n_visits <= 4:
            cn_idx = _cn_mci_indices(df)
            cn_mci = df.iloc[cn_idx]
            mci_ad = df.drop(df.index[cn_idx])

            # detect and remove reverters
            rev_cm = _detect_reverters(cn_mci, 'CN_MCI')
            rev_ma = _detect_reverters(mci_ad, 'MCI_AD')
            if rev_cm:
                reverters_cn_mci.append(cn_mci.iloc[rev_cm])
                cn_mci = cn_mci.drop(cn_mci.index[rev_cm])
            if rev_ma:
                reverters_mci_ad.append(mci_ad.iloc[rev_ma])
                mci_ad = mci_ad.drop(mci_ad.index[rev_ma])

            cn_mci.to_csv(os.path.join(dest_dir, f"{n_visits}visit_CN_MCI.csv"), index=False)
            mci_ad.to_csv(os.path.join(dest_dir, f"{n_visits}visit_MCI_AD.csv"), index=False)

        elif n_visits == 5:
            cn_idx = _cn_mci_indices(df)
            new_5_cn_mci = df.iloc[cn_idx].copy()
            new_5_mci_ad = df.drop(df.index[cn_idx]).copy()

            # detect and remove reverters
            rev_cm = _detect_reverters(new_5_cn_mci, 'CN_MCI')
            rev_ma = _detect_reverters(new_5_mci_ad, 'MCI_AD')
            if rev_cm:
                reverters_cn_mci.append(new_5_cn_mci.iloc[rev_cm])
                new_5_cn_mci = new_5_cn_mci.drop(new_5_cn_mci.index[rev_cm])
            if rev_ma:
                reverters_mci_ad.append(new_5_mci_ad.iloc[rev_ma])
                new_5_mci_ad = new_5_mci_ad.drop(new_5_mci_ad.index[rev_ma])

            new_5_cn_mci.to_csv(os.path.join(dest_dir, "original_5visit_CN_MCI.csv"), index=False)
            new_5_mci_ad.to_csv(os.path.join(dest_dir, "original_5visit_MCI_AD.csv"), index=False)

        else:
            # 6–10 visits: slice to 5 time points and augment
            sliced_cn = time_series_slicer(1, df, 5)
            sliced_ma = time_series_slicer(0, df, 5)

            # reverter check on sliced data
            rev_cm = _detect_reverters(sliced_cn, 'CN_MCI')
            rev_ma = _detect_reverters(sliced_ma, 'MCI_AD')
            if rev_cm:
                reverters_cn_mci.append(sliced_cn.iloc[rev_cm])
                sliced_cn = sliced_cn.drop(sliced_cn.index[rev_cm])
            if rev_ma:
                reverters_mci_ad.append(sliced_ma.iloc[rev_ma])
                sliced_ma = sliced_ma.drop(sliced_ma.index[rev_ma])

            if not sliced_cn.empty:
                new_5_cn_mci = pd.concat([new_5_cn_mci, sliced_cn], axis=0)
            if not sliced_ma.empty:
                new_5_mci_ad = pd.concat([new_5_mci_ad, sliced_ma], axis=0)

    # ── Save combined 5-visit datasets ────────────────────────────────────
    if not new_5_cn_mci.empty:
        new_5_cn_mci.to_csv(os.path.join(dest_dir, "5visit_CN_MCI.csv"), index=False)
    if not new_5_mci_ad.empty:
        new_5_mci_ad.to_csv(os.path.join(dest_dir, "5visit_MCI_AD.csv"), index=False)

    # ── Save reverters ────────────────────────────────────────────────────
    if reverters_cn_mci:
        pd.concat(reverters_cn_mci).to_csv(os.path.join(dest_dir, "reverters_CN_MCI.csv"), index=False)
    if reverters_mci_ad:
        pd.concat(reverters_mci_ad).to_csv(os.path.join(dest_dir, "reverters_MCI_AD.csv"), index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print("\n── Drop summary ──")
        for key in ("bmi_dropped", "mmse_dropped", "age_dropped"):
            for f, n in stats[key].items():
                if n:
                    print(f"  {f}: {n} rows ({key.replace('_', ' ')})")
        n_rev_cm = sum(len(r) for r in reverters_cn_mci) if reverters_cn_mci else 0
        n_rev_ma = sum(len(r) for r in reverters_mci_ad) if reverters_mci_ad else 0
        print(f"  Reverters saved — CN/MCI: {n_rev_cm}, MCI/AD: {n_rev_ma}")
        print(f"\n✓ Output written to {dest_dir}/")

    return stats