"""Validate data/splits.csv against a local copy of the dataset.

Usage (from anywhere):
    python check_data.py <dataset_root> [--splits PATH] [--expected-sha256 HEX]

<dataset_root> is the folder that CONTAINS the outer `chest_xray/` folder, because
paths in splits.csv look like `chest_xray/train/NORMAL/IM-0001-0001.jpeg`.

Read-only: nothing is modified. Exit code 0 = all checks passed, 1 = failure,
2 = bad arguments / unreadable input.
"""
import argparse
import hashlib
import sys
from pathlib import Path, PurePosixPath

import pandas as pd

COLUMNS = ["path", "label", "subtype", "patient_id", "orig_split", "split"]
LABELS = {"NORMAL", "PNEUMONIA"}
SPLITS = {"train", "val", "test"}
SUBTYPES = {"NORMAL": {"normal"}, "PNEUMONIA": {"bacteria", "virus"}}
MAX_LIST = 10


def normalized_sha256(path):
    """SHA-256 of the file with CRLF converted to LF (line-ending neutral)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest(), data


def to_posix(p):
    """Return the path with / separators; works for Windows- and POSIX-style input."""
    return str(p).strip().replace("\\", "/")


def usage_error(message):
    """Report invalid arguments / unreadable input and exit with code 2."""
    print(message, file=sys.stderr)
    sys.exit(2)


def show(items, label):
    for x in list(items)[:MAX_LIST]:
        print(f"    {label}: {x}")
    if len(items) > MAX_LIST:
        print(f"    ... and {len(items) - MAX_LIST} more")


def main():
    default_splits = Path(__file__).resolve().parent.parent / "data" / "splits.csv"
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset_root", help="folder containing the outer chest_xray/ folder")
    ap.add_argument("--splits", default=str(default_splits), help="path to splits.csv")
    ap.add_argument("--expected-sha256", help="expected line-ending-neutral (LF) hash")
    args = ap.parse_args()

    root = Path(args.dataset_root).expanduser()
    splits_path = Path(args.splits).expanduser()
    if not root.is_dir():
        usage_error(f"ERROR: dataset root is not a directory: {root}")
    if not splits_path.is_file():
        usage_error(f"ERROR: splits file not found: {splits_path}")
    if not (root / "chest_xray").is_dir():
        hint = ""
        if (root / "train").is_dir() and (root / "test").is_dir():
            hint = (" It looks like you passed the chest_xray folder itself;"
                    " pass its parent instead.")
        usage_error(f"ERROR: {root / 'chest_xray'} does not exist.{hint}")

    df = pd.read_csv(splits_path, dtype=str, keep_default_na=False)
    missing_cols = [c for c in COLUMNS if c not in df.columns]
    if missing_cols:
        usage_error(f"ERROR: splits.csv is missing columns {missing_cols}; found {list(df.columns)}")

    failures = []

    def check(name, bad, detail=None, label="item"):
        if len(bad):
            failures.append(name)
            print(f"[FAIL] {name}: {len(bad)}")
            show(detail if detail is not None else bad, label)
        else:
            print(f"[ OK ] {name}")

    print(f"Rows: {len(df)} | patients: {df.patient_id.nunique()} | splits file: {splits_path}")

    # --- empty cells, labels, splits, subtypes
    empty = [c for c in COLUMNS if (df[c].str.strip() == "").any()]
    check("no empty cells", empty, label="column with empty cells")
    check("labels in {NORMAL, PNEUMONIA}", df[~df.label.isin(LABELS)].label.unique(),
          label="invalid label")
    check("split in {train, val, test}", df[~df.split.isin(SPLITS)].split.unique(),
          label="invalid split")
    bad_sub = df[[s not in SUBTYPES.get(l, set()) for l, s in zip(df.label, df.subtype)]]
    check("subtype consistent with label", bad_sub, bad_sub.path.tolist(), "path")

    # --- paths: normalise separators, reject absolute / parent-escaping paths
    posix = df.path.map(to_posix)
    unsafe = [p for p in posix
              if PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts or ":" in p]
    check("paths are relative and stay inside the root", unsafe, label="path")
    check("paths start with chest_xray/ (outer copy)",
          [p for p in posix if not p.startswith("chest_xray/")], label="path")
    check("no nested chest_xray/chest_xray/ copy",
          [p for p in posix if p.startswith("chest_xray/chest_xray/")], label="path")

    # --- label agrees with the folder name in the path
    folder_mismatch = [p for p, l in zip(posix, df.label)
                       if PurePosixPath(p).parent.name != l]
    check("label matches parent folder", folder_mismatch, label="path")

    # --- duplicates
    check("no duplicate paths", posix[posix.duplicated(keep=False)].unique(), label="path")

    # --- patient overlap between splits
    n_splits = df.groupby("patient_id").split.nunique()
    leaked = n_splits[n_splits > 1]
    check("no patient in more than one split", leaked,
          [f"{pid} -> {sorted(df[df.patient_id == pid].split.unique())}" for pid in leaked.index],
          "patient")

    # --- files exist on disk
    unsafe_set = set(unsafe)
    missing = [p for p in posix if p not in unsafe_set and not root.joinpath(*p.split("/")).is_file()]
    check("every path exists under the dataset root", missing, label="missing")

    # --- line-ending-neutral hash
    digest, data = normalized_sha256(splits_path)
    crlf_digest = hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest()
    print(f"splits.csv sha256 (LF-normalised): {digest}")
    print(f"splits.csv sha256 (CRLF form, as recorded in data/README.md): {crlf_digest}")
    if args.expected_sha256:
        ok = args.expected_sha256.lower() in (digest, crlf_digest)
        check("hash matches --expected-sha256", [] if ok else ["mismatch"], label="hash")

    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
    if failures:
        print(f"\nFAIL: {len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("\nPASS: all checks passed")


if __name__ == "__main__":
    main()
