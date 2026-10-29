# Dataset: Chest X-Ray Images (Pneumonia)

## Provenance
- **Source (downloaded from):** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
- **Original host:** https://data.mendeley.com/datasets/rscbjbr9sj/2 (Version 2, published 2018-01-06)
- **Paper:** Kermany et al., Cell 2018, http://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
- **Downloaded:** 19/10/2026
- **Zip sha256:** f569fe885b0f921e836f3d6bcc8d7b3442f5e0ca4db4533d06b8cf25d2114ea1
- **Licence:** CC BY 4.0 (stated in the dataset description; attribution required).
  Kaggle licence field reads "Other (specified in description)".
- **Attribution:** Data: Kermany et al., Cell 2018, CC BY 4.0.
- **Task:** binary classification, NORMAL vs PNEUMONIA
- **Proposed label mapping:** NORMAL=0, PNEUMONIA=1

## Label provenance (from the source description)
Chest X-rays (anterior-posterior) of pediatric patients aged 1-5 from Guangzhou
Women and Children's Medical Center, taken as part of routine care. Low-quality
or unreadable scans were removed, diagnoses were graded by two expert
physicians, and the evaluation set was also checked by a third. Pneumonia
filenames also encode bacteria/virus.

## Contents
5,856 unique JPEG images, 2,790 patient IDs parsed from filenames (about 2.1
images per patient, max 31). The images are NOT stored in this repository.
Download them from the source above.

## Data-quality findings
Findings are reported, not silently fixed. No files were deleted.

1. **Duplicate packaging:** the zip contains a byte-identical nested copy
   (`chest_xray/chest_xray/`, 5,856 files, all hashes match the outer copy) and a
   `__MACOSX` folder (5,856 macOS resource-fork stubs with a .jpeg extension,
   not real images) plus 16 `.DS_Store` files. All were excluded from analysis,
   none deleted. Counting everything in the zip gives 17,568 files; the real
   dataset is 5,856 images.
2. **Corrupt files:** 0 of 5,856 failed to open and load.
3. **Exact duplicates:** 32 redundant files (30 hashes) in the outer copy. All are
   same-patient and same-split. No hash appears in more than one split, and no
   identical image has conflicting labels.
4. **Original validation set is tiny:** 16 images (8 per class).
5. **Original split is not patient-level:** 170 pneumonia patient IDs appear in
   both train and test (320 of 390 test pneumonia images). 94 normal IDs appear
   in more than one split, but these pair two filename series (`IM-` and
   `NORMAL2-IM-`) and may be ID collisions, so leakage there is unconfirmed.
6. **Image size varies and differs by class:** width 384-2,916 px, height
   127-2,713 px. Median 1654x1323 px (NORMAL) vs 1160x776 px (PNEUMONIA).
7. **Colour mode varies and differs by class:** 5,573 grayscale, 283 RGB (4.8%).
   All 283 RGB images are PNEUMONIA.
8. **Shortcut risk:** a single height threshold classifies 84.8% of images
   correctly (width: 82.4%) vs a 73.0% majority-class baseline. Thresholds were
   chosen on the full dataset (no held-out set), so treat this as indicative.
   A model may exploit acquisition/packaging artifacts instead of anatomy.
9. **Image count differs from the source description:** the Kaggle description
   states 5,863 images; 5,856 unique image files were found in the downloaded
   copy (difference of 7, cause not determined).

## Recommended split
Patient-level 70/15/15, grouped by patient ID parsed from filenames, with the
`IM-` and `NORMAL2-IM-` series treated as one group (conservative). Seed 42.
File: `data/splits.csv` (columns: path, label, subtype, patient_id, orig_split, split).
sha256 of `splits.csv`: `0eacd0e7ed1e0122ead2110faf708cc27575c9fbfec943da9f2d88ef7e0e2310`
(rebuilding from the raw dataset reproduced this hash).

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| train | 1,109 | 2,991 | 4,100 |
| val | 237 | 641 | 878 |
| test | 237 | 641 | 878 |

Pneumonia-to-normal ratio is about 2.7:1 in every split. `verify_splits.py`
confirms no patient appears in more than one split.

Trade-off: results are not directly comparable with published numbers that use
the original split.

## Loader recommendations
- Read paths and labels from `data/splits.csv`, not from the original folders.
- Use only the outer `chest_xray/` folder (paths in `splits.csv` start with `chest_xray/`).
- Convert every image to one channel count, resize every image to one fixed size
  with one method, and apply identical preprocessing to train, val and test.
- Report per-class recall alongside accuracy.

## Limitations
Single site, pediatric only; class imbalance (about 2.7:1); no clinical metadata
to cross-check labels; patient IDs inferred from filenames; image-size and
colour-mode artifacts correlate with the label.

## Reproduce
Run from `project-1-healthcare/scripts/`. `inventory.py` ignores `__MACOSX`,
`.DS_Store` and the nested copy is excluded by `add_patient_ids.py`.

    python inventory.py <path to unzipped dataset>
    python add_patient_ids.py
    python make_splits.py
    python verify_splits.py     (must print PASS and the sha256 above)
    python eda.py <path to unzipped dataset>

EDA figures are written to `data/eda/`.
