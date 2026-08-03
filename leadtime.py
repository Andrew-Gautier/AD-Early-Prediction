import numpy as np
import pandas as pd
import os
from feature_engineering import create_delta_features, preprocess_data

nan = np.nan
_LONG_COLS = [
    'NACCBMI', 'NACCMMSE', 'NACCGDS', 'CDRSUM', 'TOBAC30',
    'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
    'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL',
    'hearing', 'vision',
    'NACCLIVS', 'ALCOHOL', 'COMMUN', 'months_since_baseline'
]
def eval_if_str(x):
    if isinstance(x, str):
        return eval(x)
    return x

# unpack one row of original features into a df of engineered features 
#    with every possible truncation being its own row.
# return # of truncations and transformed df.
def transform(row, progression_type, mask_length=0):
    num_trunc = row['n_visits']-1-mask_length
    out = []
    for length in range(2, num_trunc+2):
        new_row = {}
        for col in row.index:
            if col in _LONG_COLS:
                if isinstance(row[col], list):
                    new_row[col] = row[col][:length]
                else:
                    new_row[col] = np.nan
            else: 
                new_row[col] = row[col]
        out.append(new_row)
    df = pd.DataFrame(out)
    df = create_delta_features(df)
    df,_,_ = preprocess_data(df, progression_type)
    # print(df.shape)
    # print(df.head())
    return num_trunc, df.drop(columns=['target'])


# ═══════════════════════════════════════════════════════════════════════════════
#  LEAD-TIME ANALYSIS  
# ═══════════════════════════════════════════════════════════════════════════════

def run_leadtime(
        file_path, # csv df address
        dest_dir,
        model,
        progression_type, # 'CN' or 'AD'
        mask_length = 0, # mask this many latest visits for negative control group
):
    os.makedirs(dest_dir, exist_ok=True)
    csv=pd.read_csv(file_path)

    # parse the df
    for col in _LONG_COLS: 
        csv[col] = csv[col].apply(eval_if_str)

    # separate by progression status
    p_df = csv[csv['Prog_ID']==1]
    np_df = csv[csv['Prog_ID']==0]

    # PROGRESSORS
    # record model prediction on every truncated visit segment to this id's csv row, 
    #   starting with the shortest segment (2 visits).
    prob_csv = []
    pred_csv = []
    for ind,row in p_df.iterrows():
        # print(f"working on {ind}")
        new_row_prob = {}
        new_row_pred = {}
        new_row_prob['ID']=row['ID']
        new_row_pred['ID']=row['ID']
        new_row_prob['months_since_baseline']=row['months_since_baseline'] 
        new_row_pred['months_since_baseline']=row['months_since_baseline']
        new_row_prob['Progression']=row['Progression']
        new_row_pred['Progression']=row['Progression']

        n,df = transform(row,progression_type)
        x= df.values
        prob = model.predict_proba(x)[:, 1]
        pred = model.predict(x)
        for i in range(n):
            new_row_prob[i]=prob[i]
            new_row_pred[i]=pred[i]
        prob_csv.append(new_row_prob)
        pred_csv.append(new_row_pred)

    prob_path = os.path.join(dest_dir,'lead_time_probabilities.csv')
    pred_path = os.path.join(dest_dir,'lead_time_predictions.csv')
    pd.DataFrame(prob_csv).to_csv(prob_path, index=False)
    pd.DataFrame(pred_csv).to_csv(pred_path, index=False)

    ## NON-PROGRESSORS
    prob_csv = []
    pred_csv = []
    for ind,row in np_df.iterrows():
        new_row_prob = {}
        new_row_pred = {}
        new_row_prob['ID']=row['ID']
        new_row_pred['ID']=row['ID']
        new_row_prob['months_since_baseline']=row['months_since_baseline'] 
        new_row_pred['months_since_baseline']=row['months_since_baseline']
        new_row_prob['Progression']=row['Progression']
        new_row_pred['Progression']=row['Progression']

        if row['n_visits']<mask_length+2:
            continue

        n,df = transform(row,progression_type,mask_length)
        x= df.values
        prob = model.predict_proba(x)[:, 1]
        pred = model.predict(x)
        for i in range(n):
            new_row_prob[i]=prob[i]
            new_row_pred[i]=pred[i]
        prob_csv.append(new_row_prob)
        pred_csv.append(new_row_pred)

    prob_path = os.path.join(dest_dir,'control_probabilities.csv')
    pred_path = os.path.join(dest_dir,'control_predictions.csv')
    pd.DataFrame(prob_csv).to_csv(prob_path, index=False)
    pd.DataFrame(pred_csv).to_csv(pred_path, index=False)


######### TESTING #############
import joblib

m=joblib.load('AD-Early-Prediction\experiments\experiment_nonimputedseeds_test\pooled_MCI_AD_AD_seed107420369_model_AD.pkl')

nan=np.nan
run_leadtime(
    'AD-Early-Prediction\datasets\Dataset_v2_1\lead_time_MCI_AD.csv',
    'lead_time',
    m,
    'AD'
)
###########################################