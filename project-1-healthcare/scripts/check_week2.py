import hashlib
import sys
from pathlib import Path

import pandas as pd

# sha256 of splits.csv after converting CRLF to LF, so it is identical
# on Windows, Linux and macOS regardless of line endings
EXPECTED_SHA256 = "5cd9a952286fc4f40415c8fd78d2042f82755f31d74d6c4b90f6ea49d787a16f"
EXPECTED_COUNTS = {("train", "NORMAL"): 1109, ("train", "PNEUMONIA"): 2991,
                   ("val", "NORMAL"): 237, ("val", "PNEUMONIA"): 641,
                   ("test", "NORMAL"): 237, ("test", "PNEUMONIA"): 641}
REQUIRED_COLUMNS = ["path", "label", "subtype", "patient_id", "orig_split", "split"]
VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {"NORMAL", "PNEUMONIA"}

SPLITS_CSV = Path(__file__).resolve().parent.parent / "data" / "splits.csv"


def sha256_lf(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def schema_errors(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing column(s): {', '.join(missing)} (found: {', '.join(df.columns)})"]
    errors = []
    for col in REQUIRED_COLUMNS:
        n = int(df[col].isna().sum())
        if n:
            errors.append(f"column '{col}' has {n} empty value(s)")
    for col, valid in [("split", VALID_SPLITS), ("orig_split", VALID_SPLITS),
                       ("label", VALID_LABELS)]:
        bad = sorted(set(df[col].dropna()) - valid)
        if bad:
            errors.append(f"column '{col}' has unexpected value(s): {bad}; "
                          f"allowed: {sorted(valid)}")
    return errors


def main():
    if not SPLITS_CSV.is_file():
        print(f"FAIL: {SPLITS_CSV} not found")
        return 1
    try:
        s = pd.read_csv(SPLITS_CSV)
    except Exception as e:
        print(f"FAIL: could not parse {SPLITS_CSV.name} as CSV: {e}")
        return 1

    errors = schema_errors(s)
    if errors:
        for e in errors:
            print(f"[FAIL] schema: {e}")
        print("RESULT: FAIL")
        return 1

    digest = sha256_lf(SPLITS_CSV)
    counts = s.groupby(["split", "label"]).size().to_dict()
    cross = int((s.groupby("patient_id").split.nunique() > 1).sum())

    print("splits.csv sha256 (LF-normalised):", digest)
    print("Rows:", len(s))
    checks = {
        "schema: required columns and values valid": True,
        "sha256 matches recorded value": digest == EXPECTED_SHA256,
        "counts match README table": counts == EXPECTED_COUNTS,
        "no duplicate paths": int(s.path.duplicated().sum()) == 0,
        "no patient in more than one split": cross == 0,
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
