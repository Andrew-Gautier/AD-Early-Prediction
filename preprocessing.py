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
        # Ensure it's a valid list format
        row = row.replace("nan", "null")  # Replace 'nan' string with 'null' for JSON compatibility
        values = json.loads(row)  # Parse JSON-formatted string into a list

        # Convert 'None' (from JSON) into np.nan for numerical operations
        return [np.nan if x is None else float(x) for x in values]
    
    except (ValueError, TypeError, json.JSONDecodeError):
        return row  # If it can't be converted, return the original
    
def impute_mmse(df, covariates):
    # Convert MMSE strings to lists of floats
    mmse_lists = df['MMSE'].apply(convert_to_real_nan)
    
    # Create a matrix of MMSE values (subjects x timepoints)
    mmse_matrix = np.array([x for x in mmse_lists])
    covariate_matrix = []
    for col in covariates:
        # Check if the column contains string representations of lists
        if df[col].dtype == 'object' and df[col].iloc[0].startswith('['):
            # Convert list-type covariates
            col_data = df[col].apply(convert_to_real_nan)
            # For list columns, we'll take all timepoints as separate covariates
            col_matrix = np.array([x for x in col_data])
            covariate_matrix.append(col_matrix)
        else:
            # For non-list columns, reshape to 2D array
            col_matrix = df[col].values.reshape(-1, 1)
            covariate_matrix.append(col_matrix)
    # Combine all covariate matrices horizontally
    covariate_data = np.hstack([x if len(x.shape) > 1 else x.reshape(-1, 1) for x in covariate_matrix])   
    # Initialize imputer
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=20,
        random_state=42,
        initial_strategy='mean'
    )
    # Combine MMSE and covariates for imputation
    combined_data = np.hstack([mmse_matrix, covariate_data])
    # Perform imputation
    imputed_data = imputer.fit_transform(combined_data)
    # Extract imputed MMSE values
    imputed_mmse = imputed_data[:, :mmse_matrix.shape[1]]
    # Process imputed values:
    # 1. Cap values at 30.0
    # 2. Round to 2 decimal places
    # 3. Convert to regular float lists
    processed_mmse = []
    for row in imputed_mmse:
        processed_row = [min(round(float(val), 2), 30.0) for val in row]
        processed_mmse.append(processed_row)
    
    # Update the DataFrame with processed imputed values
    df['MMSE'] = processed_mmse
    
    return df
