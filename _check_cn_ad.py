"""Temporary script: check if CN->AD or CN->MCI->AD patients exist in raw data."""
import pandas as pd, ast, os
from collections import Counter

# 1. What patterns exist in pooled_CN progressors?
df = pd.read_csv("datasets/Dataset_pooled/pooled_CN.csv")
progs = df[df["Prog_ID"] == 1]
reach_ad = progs["Progression"].apply(lambda x: 2 in ast.literal_eval(x)).sum()
reach_mci_only = progs["Progression"].apply(
    lambda x: 1 in ast.literal_eval(x) and 2 not in ast.literal_eval(x)
).sum()
print(f"Pooled CN progressors: {len(progs)}")
print(f"  reaching AD (2): {reach_ad}")
print(f"  reaching MCI only (1): {reach_mci_only}")

# Show a few example progressions
print("\nSample progressor Progression tuples:")
for _, row in progs.head(10).iterrows():
    print(f"  {row['Progression']}")

# 2. Scan ALL raw cohort files for CN-starting patients that reach AD
print("\n=== Raw cohorts: CN-starting patients reaching AD (label 2) ===")
found = 0
for n in range(2, 11):
    fpath = f"datasets/Raw_Cohorts/{n}visit_combined.csv"
    if not os.path.exists(fpath):
        continue
    raw = pd.read_csv(fpath)
    for _, row in raw.iterrows():
        prog = ast.literal_eval(str(row["Progression"]))
        if 0 in prog and 2 in prog:
            found += 1
            print(f"  {n}visit: ID={row['ID']}, Prog_ID={row['Prog_ID']}, Progression={prog}")
if found == 0:
    print("  NONE FOUND in any raw cohort file.")

# 3. Also check: what Prog_ID values do CN-starting patients have?
print("\n=== Prog_ID distribution for CN-starting patients (raw) ===")
for n in range(2, 11):
    fpath = f"datasets/Raw_Cohorts/{n}visit_combined.csv"
    if not os.path.exists(fpath):
        continue
    raw = pd.read_csv(fpath)
    cn_rows = []
    for _, row in raw.iterrows():
        prog = ast.literal_eval(str(row["Progression"]))
        if 0 in prog:
            cn_rows.append(row)
    if cn_rows:
        cn_df = pd.DataFrame(cn_rows)
        max_labels = cn_df["Progression"].apply(lambda x: max(ast.literal_eval(x)))
        counts = Counter(max_labels)
        print(f"  {n}visit: {len(cn_df)} CN-starting, max-label distribution: {dict(counts)}")
