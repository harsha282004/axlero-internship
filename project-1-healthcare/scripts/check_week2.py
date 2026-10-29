import hashlib
import sys
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "0eacd0e7ed1e0122ead2110faf708cc27575c9fbfec943da9f2d88ef7e0e2310"
EXPECTED_COUNTS = {("train", "NORMAL"): 1109, ("train", "PNEUMONIA"): 2991,
                   ("val", "NORMAL"): 237, ("val", "PNEUMONIA"): 641,
                   ("test", "NORMAL"): 237, ("test", "PNEUMONIA"): 641}

# resolved from the script location, so it works from any working directory
SPLITS_CSV = Path(__file__).resolve().parent.parent / "data" / "splits.csv"


def main():
    if not SPLITS_CSV.is_file():
        print(f"FAIL: {SPLITS_CSV} not found")
        return 1

    digest = hashlib.sha256(SPLITS_CSV.read_bytes()).hexdigest()
    s = pd.read_csv(SPLITS_CSV)
    counts = s.groupby(["split", "label"]).size().to_dict()
    cross = int((s.groupby("patient_id").split.nunique() > 1).sum())

    print("splits.csv sha256:", digest)
    print("Rows:", len(s))
    checks = {
        "sha256 matches recorded value": digest == EXPECTED_SHA256,
        "counts match README table": counts == EXPECTED_COUNTS,
        "no duplicate paths": int(s.path.duplicated().sum()) == 0,
        "no patient in more than one split": cross == 0,
        "labels are NORMAL and PNEUMONIA": sorted(s.label.unique()) == ["NORMAL", "PNEUMONIA"],
    }
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    if not all(checks.values()):
        print("RESULT: FAIL")
        return 1
    print("RESULT: UNCHANGED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
