import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
import os
import json

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