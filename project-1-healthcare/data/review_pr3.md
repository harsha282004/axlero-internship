# Review notes: PR #3 (CNN training and evaluation)

Reviewer: Sri Chidvilas Govindarajula (dataset owner). Date: 2026-10-28.

## Verified
- train.py and finetune.py read data/splits.csv; finetune.py also checks for
  duplicate image paths.
- Fine-tuned test confusion matrix [[222, 15], [13, 628]] sums to 878 with rows
  of 237 (NORMAL) and 641 (PNEUMONIA), matching the proposed patient-level test
  split (the original test split would be 234 / 390).
- Baseline NORMAL row is 190 + 47 = 237; PNEUMONIA row not yet inspected in full.
- Class index 0 = NORMAL (normal_recall = recall[0]), consistent with the
  proposed mapping NORMAL=0, PNEUMONIA=1.
- Every image is converted to RGB and resized to one fixed square size, in both
  train and eval transforms.
- Per-class precision, recall, F1 and a confusion matrix are reported.
- No dataset images, zips or model weights are committed.

## To confirm
- No train_test_split or random_split in the code
- Train and val counts (expected 4,100 and 878)
- Best model chosen on validation, not test

## Caveats
- A square resize stretches images, and median aspect ratio differs by class
  (about 1.25 NORMAL vs 1.49 PNEUMONIA), so shape cues may remain.
- Results are on a patient-level split and are not comparable with published
  numbers on the original Kaggle split.
