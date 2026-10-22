import hashlib
import pandas as pd

EXPECTED = "0eacd0e7ed1e0122ead2110faf708cc27575c9fbfec943da9f2d88ef7e0e2310"
EXPECTED_COUNTS = {("train", "NORMAL"): 1109, ("train", "PNEUMONIA"): 2991,
                   ("val", "NORMAL"): 237, ("val", "PNEUMONIA"): 641,
                   ("test", "NORMAL"): 237, ("test", "PNEUMONIA"): 641}

raw = open("../data/splits.csv", "rb").read()
h = hashlib.sha256(raw).hexdigest()
print("splits.csv sha256:", h)
print("Hash matches recorded value:", h == EXPECTED)

s = pd.read_csv("../data/splits.csv")
counts = s.groupby(["split", "label"]).size().to_dict()
print("Counts match README table:", counts == EXPECTED_COUNTS)
print("Rows:", len(s), "| duplicate paths:", s.path.duplicated().sum())
print("Patients in >1 split:", (s.groupby("patient_id").split.nunique() > 1).sum())
print("Labels present:", sorted(s.label.unique()))
ok = (h == EXPECTED and counts == EXPECTED_COUNTS
      and not s.path.duplicated().any()
      and (s.groupby("patient_id").split.nunique() > 1).sum() == 0)
print("RESULT:", "UNCHANGED" if ok else "CHECK: something differs")
