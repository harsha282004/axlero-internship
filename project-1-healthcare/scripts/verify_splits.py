from pathlib import Path
import csv
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "splits.csv"
DATASET_ROOT = PROJECT_ROOT / "chest_xray"

REQUIRED_COLUMNS = {"path", "label", "split", "patient_id"}
VALID_SPLITS = {"train", "val", "test"}
VALID_LABELS = {"NORMAL", "PNEUMONIA"}


def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Splits file not found: {CSV_PATH}")

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")

    with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file is empty.")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no data rows.")

    split_counts = Counter()
    label_counts = Counter()
    class_distribution = defaultdict(Counter)
    patient_splits = defaultdict(set)
    seen_paths = set()

    missing_paths = []
    duplicate_paths = []
    invalid_rows = []

    for row_number, row in enumerate(rows, start=2):
        relative_path = (row.get("path") or "").strip()
        label = (row.get("label") or "").strip()
        split = (row.get("split") or "").strip()
        patient_id = (row.get("patient_id") or "").strip()

        if not all([relative_path, label, split, patient_id]):
            invalid_rows.append(row_number)
            continue

        if label not in VALID_LABELS or split not in VALID_SPLITS:
            invalid_rows.append(row_number)
            continue

        split_counts[split] += 1
        label_counts[label] += 1
        class_distribution[split][label] += 1

        if relative_path in seen_paths:
            duplicate_paths.append(relative_path)

        seen_paths.add(relative_path)
        patient_splits[patient_id].add(split)

        image_path = PROJECT_ROOT / Path(relative_path)
        if not image_path.is_file():
            missing_paths.append(relative_path)

    print(f"Total rows: {len(rows)}")

    print("\nSplit distribution:")
    for split in sorted(VALID_SPLITS):
        print(f"{split}: {split_counts[split]}")

    print("\nClass distribution:")
    for split in sorted(VALID_SPLITS):
        print(f"\n{split}:")
        for label in sorted(VALID_LABELS):
            print(f"  {label}: {class_distribution[split][label]}")

    print("\nLabel distribution:")
    for label in sorted(VALID_LABELS):
        print(f"{label}: {label_counts[label]}")

    patient_overlaps = {
        patient_id: splits
        for patient_id, splits in patient_splits.items()
        if len(splits) > 1
    }

    print(f"\nUnique patient IDs: {len(patient_splits)}")
    print(f"Patient overlaps: {len(patient_overlaps)}")
    print(f"Duplicate paths: {len(duplicate_paths)}")
    print(f"Invalid rows: {len(invalid_rows)}")
    print(f"Missing image paths: {len(missing_paths)}")

    if patient_overlaps or duplicate_paths or invalid_rows or missing_paths:
        raise SystemExit("\nFAIL: Dataset verification failed.")

    print("\nPASS: Dataset verification completed successfully.")


if __name__ == "__main__":
    main()
