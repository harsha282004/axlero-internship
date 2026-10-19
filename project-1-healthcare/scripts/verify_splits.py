import hashlib, sys
import pandas as pd

df = pd.read_csv("../data/splits.csv")
print("Rows:", len(df))
print(df.groupby(["split", "label"]).size().unstack(), "\n")

n = df.groupby("patient_id").split.nunique()
bad = n[n > 1]
print("Patients in >1 split:", len(bad))
print("Duplicate paths:", df.path.duplicated().sum())
print("splits.csv sha256:", hashlib.sha256(open("../data/splits.csv", "rb").read()).hexdigest())

if len(bad) or df.path.duplicated().any():
    sys.exit("FAIL: leakage or duplicate paths found")
print("PASS")
