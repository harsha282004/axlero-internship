import hashlib
import sys
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "splits.csv"
df = pd.read_csv(CSV)
print("Rows:", len(df))
print(df.groupby(["split", "label"]).size().unstack(), "\n")

n = df.groupby("patient_id").split.nunique()
bad = n[n > 1]
print("Patients in >1 split:", len(bad))
print("Duplicate paths:", df.path.duplicated().sum())
digest = hashlib.sha256(CSV.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
print("splits.csv sha256 (LF-normalised):", digest)

if len(bad) or df.path.duplicated().any():
    sys.exit("FAIL: leakage or duplicate paths found")
print("PASS")
