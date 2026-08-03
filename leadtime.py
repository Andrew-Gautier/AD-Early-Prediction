import numpy as np
import pandas as pd
import os
from feature_engineering import create_delta_features, preprocess_data

nan = np.nan
_LONG_COLS = [
    'NACCBMI', 'NACCMMSE', 'NACCGDS', 'CDRSUM', 'TOBAC30',
    'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE',
    'MEALPREP', 'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL',
    'NACCLIVS', 'COMMUN', 'ALCOHOL', 'months_since_baseline'
]
def eval_if_str(x):
    if isinstance(x, str):
        return eval(x)
    return x

# unpack one row of original features into a df of engineered features 
#    with every possible truncation being its own row.
# return # of truncations and transformed df.
def transform(row, progression_type):
    num_trunc = row['n_visit']-1
    out = []
    for length in range(2, num_trunc+2):
        new_row = {}
        for col in row.columns:
            if col in _LONG_COLS:
                new_row[col] = row[col].apply(
                    lambda x: x[:length] if isinstance(x, list) else np.nan)
            else: new_row[col] = row[col]
        out.append(new_row)
    df = pd.DataFrame(out)
    df,_,_ = create_delta_features(df)
    df,_,_ = preprocess_data(df, progression_type)

    return num_trunc, df.drop(columns=['target'])


# ═══════════════════════════════════════════════════════════════════════════════
#  LEAD-TIME ANALYSIS  
# ═══════════════════════════════════════════════════════════════════════════════

def run_leadtime(
        file_path, # csv df address
        dest_dir,
        model,
        progression_type,
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
    for _,row in p_df.iterrows():
        new_row_prob = {}
        new_row_pred = {}
        new_row_prob['ID']=row['ID']
        new_row_pred['ID']=row['ID']
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

