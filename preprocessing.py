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
    # Check if the DataFrame contains the 'Prog_ID' column
    if 'Prog_ID' in df.columns:
        # Count the number of unique values in the 'Prog_ID' column
        progressors = df['Prog_ID'] == 1
        non_progressors = df['Prog_ID'] == 0
        
        print(f"File: {file_path} - Progressors: {progressors.sum()}, Non-Progressors: {non_progressors.sum()}")
    mci_count = 0
    cn_to_mci = 0
    mci_to_ad = 0
    for ids, progression in enumerate(df['Progression']):
        if isinstance(progression, str):
            # Convert the string representation of the list to an actual list
            progression = eval(progression)
        if all(val == 1 for val in progression):
            mci_count += 1
        # CN to MCI 
        if progression[0] == 0 in progression and df.loc[ids, 'Prog_ID'] == 1:
            cn_to_mci += 1
        # MCI to AD
        if progression[0] == 1 in progression and df.loc[ids, 'Prog_ID'] == 1:
            mci_to_ad += 1
    print(f"Stable Mci in {file_path} : {mci_count}")
    print (f"Stable CN in {file_path} : {non_progressors.sum() - mci_count}")
    print (f"CN to MCI in {file_path} : {cn_to_mci}")
    print (f"MCI to AD in {file_path} : {mci_to_ad}")

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
    """
    Collapse a response_sequences DataFrame down to one row per unique pattern,
    sorted by frequency descending.

    Parameters
    ----------
    sequences_df : output of build_response_sequences
    save_csv     : optional path to save the summary

    Returns
    -------
    DataFrame with columns [Responses, Count, Coding, Pct]
    """
    summary = (
        sequences_df
        .groupby(["Responses", "Coding"], as_index=False)["Count"]
        .first()                          # Count is already per-pattern
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
    parent_dir = os.path.dirname(current_dir) 
    target_file = os.path.join(parent_dir, 'investigator_nacc67.csv')
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