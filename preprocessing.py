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
    
