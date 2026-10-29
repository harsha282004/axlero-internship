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

## Confirmed (code and report)
- Split counts: train 4,100 / val 878 / test 878 in experiment_report.md, matching
  data/splits.csv. Baseline and fine-tuned test confusion matrices have rows of
  237 (NORMAL) and 641 (PNEUMONIA), summing to 878.
- No random splitting: no train_test_split or random_split anywhere in src/.
  shuffle=True is only on the training loader; validation and test use shuffle=False.
- Validation-based model selection: train.py (line 425) and finetune.py (line 697)
  save a checkpoint only when validation loss improves (val_loss < best_val_loss).
  Test loss is not used for selection.
- Both scripts read data/splits.csv; every image is converted to RGB and resized to
  one fixed size in train and eval transforms; no images or weights committed.

## Caveats
- Square resize stretches images; median aspect ratio differs by class.
- Both models trained for 2 epochs in a single run; improvement is indicative.
- Results are on a patient-level split and are not comparable with published
  numbers on the original Kaggle split.