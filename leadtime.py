import numpy as np
import pandas as pd
import os
from feature_engineering import create_delta_features, preprocess_data
import ast
import matplotlib.pyplot as plt

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




def _parse_list(s):
    """Parse stringified list columns (months_since_baseline, Progression)."""
    if isinstance(s, str):
        return ast.literal_eval(s)
    return s


def _pred_cols(df):
    """Return the truncation-index prediction columns ('0', '1', ...), sorted numerically."""
    cols = [c for c in df.columns if c not in ('ID', 'months_since_baseline', 'Progression')]
    return sorted(cols, key=lambda x: int(x))


def load_leadtime_results(base_dir):
    """Load progressor (lead_time_*) and control (control_*) prediction csvs for a cohort dir."""
    prog_pred = pd.read_csv(os.path.join(base_dir, 'lead_time_predictions.csv'))
    ctrl_pred = pd.read_csv(os.path.join(base_dir, 'control_predictions.csv'))
    for df in (prog_pred, ctrl_pred):
        df['months_since_baseline'] = df['months_since_baseline'].apply(_parse_list)
        df['Progression'] = df['Progression'].apply(_parse_list)
    return prog_pred, ctrl_pred


def analyze_progressors(prog_pred):
    """For each progressor, find the first truncation at which the model raised an alarm (pred==1),
    and compute the resulting lead time relative to the true conversion visit."""
    pred_cols = _pred_cols(prog_pred)
    records = []
    for _, row in prog_pred.iterrows():
        months = row['months_since_baseline']
        conversion_month = months[-1]
        baseline_month = months[0]

        first_alarm_idx = None
        for c in pred_cols:
            val = row[c]
            if pd.notna(val) and val == 1:
                first_alarm_idx = int(c)
                break

        if first_alarm_idx is None:
            status = 'missed'
            lead_time = np.nan
            alarm_month = np.nan
        else:
            alarm_month = months[first_alarm_idx + 1]
            lead_time = conversion_month - alarm_month
            status = 'detected_early' if lead_time > 0 else 'detected_at_conversion'

        records.append({
            'ID': row['ID'],
            'baseline_month': baseline_month,
            'conversion_month': conversion_month,
            'alarm_month': alarm_month,
            'lead_time_months': lead_time,
            'status': status,
        })
    return pd.DataFrame(records)


def analyze_controls(ctrl_pred):
    """For each control (non-progressor), compute whether/how often the model raised a false alarm."""
    pred_cols = _pred_cols(ctrl_pred)
    records = []
    for _, row in ctrl_pred.iterrows():
        vals = [row[c] for c in pred_cols if pd.notna(row[c])]
        n_alarms = int(sum(v == 1 for v in vals))
        records.append({
            'ID': row['ID'],
            'n_visits_predicted': len(vals),
            'n_false_alarms': n_alarms,
            'any_false_alarm': n_alarms > 0,
            'false_alarm_rate': n_alarms / len(vals) if vals else np.nan,
        })
    return pd.DataFrame(records)


def summarize_leadtime(prog_df, ctrl_df, label=""):
    """Print sensitivity/specificity and lead-time stats for one cohort."""
    n_prog = len(prog_df)
    n_detected = int((prog_df['status'] != 'missed').sum())
    n_early = int((prog_df['status'] == 'detected_early').sum())
    n_at_conv = int((prog_df['status'] == 'detected_at_conversion').sum())
    n_missed = int((prog_df['status'] == 'missed').sum())
    sensitivity = n_detected / n_prog if n_prog else np.nan

    n_ctrl = len(ctrl_df)
    n_false_alarm = int(ctrl_df['any_false_alarm'].sum())
    specificity = 1 - n_false_alarm / n_ctrl if n_ctrl else np.nan

    lead_times = prog_df.loc[prog_df['status'] == 'detected_early', 'lead_time_months']

    print(f"===== {label} Lead-Time Analysis =====")
    print(f"Progressors: {n_prog}")
    print(f"  Detected (any point):            {n_detected:3d} ({sensitivity:.1%})")
    print(f"    - Detected early (pre-conversion): {n_early:3d} ({n_early/n_prog:.1%})")
    print(f"    - Detected only at conversion:     {n_at_conv:3d} ({n_at_conv/n_prog:.1%})")
    print(f"  Missed entirely:                  {n_missed:3d} ({n_missed/n_prog:.1%})")
    if len(lead_times):
        print(f"  Lead time (months) among early detections: "
              f"mean={lead_times.mean():.1f}, median={lead_times.median():.1f}, "
              f"min={lead_times.min():.1f}, max={lead_times.max():.1f}")
    print(f"\nControls (non-progressors): {n_ctrl}")
    print(f"  False alarms (>=1 visit): {n_false_alarm:3d} ({n_false_alarm/n_ctrl:.1%})")
    print(f"  Specificity: {specificity:.1%}")

    return {
        'n_prog': n_prog, 'n_detected': n_detected, 'n_early': n_early,
        'n_at_conv': n_at_conv, 'n_missed': n_missed, 'sensitivity': sensitivity,
        'n_ctrl': n_ctrl, 'n_false_alarm': n_false_alarm, 'specificity': specificity,
        'lead_times': lead_times,
    }


def plot_leadtime_summary(prog_df, ctrl_df, label=""):
    """Plot detection outcome breakdown, lead-time distribution, and control false-alarm rate."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    status_order = ['detected_early', 'detected_at_conversion', 'missed']
    status_counts = prog_df['status'].value_counts().reindex(status_order, fill_value=0)
    axes[0].bar(['Early', 'At conversion', 'Missed'], status_counts.values,
                color=['#2ca02c', '#ff7f0e', '#d62728'])
    axes[0].set_title(f"{label}: Progressor Detection Outcomes")
    axes[0].set_ylabel("Number of subjects")

    lead_times = prog_df.loc[prog_df['status'] == 'detected_early', 'lead_time_months']
    if len(lead_times):
        axes[1].hist(lead_times, bins=15, color='#1f77b4', edgecolor='white')
    axes[1].set_title(f"{label}: Lead Time Distribution")
    axes[1].set_xlabel("Months before conversion")
    axes[1].set_ylabel("Count")

    fa_counts = ctrl_df['any_false_alarm'].value_counts().reindex([False, True], fill_value=0)
    axes[2].bar(['No alarm', 'False alarm'], fa_counts.values, color=['#2ca02c', '#d62728'])
    axes[2].set_title(f"{label}: Control False Alarms")
    axes[2].set_ylabel("Number of subjects")

    plt.tight_layout()
    plt.show()

######### TESTING #############
# import joblib

# m=joblib.load('AD-Early-Prediction\experiments\experiment_nonimputedseeds_test\pooled_MCI_AD_AD_seed107420369_model_AD.pkl')

# nan=np.nan
# run_leadtime(
#     'AD-Early-Prediction\datasets\Dataset_v2_1\lead_time_MCI_AD.csv',
#     'lead_time',
#     m,
#     'AD'
# )
###########################################