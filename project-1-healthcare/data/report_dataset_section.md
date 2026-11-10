# Dataset section for the final report

## Source and licence
Chest X-Ray Images (Pneumonia), Kermany et al., Cell 2018. Downloaded from Kaggle
(https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on 2026-10-19;
original host https://data.mendeley.com/datasets/rscbjbr9sj/2 (Version 2). Licence: CC BY 4.0
(attribution required). Zip sha256: f569fe885b0f921e836f3d6bcc8d7b3442f5e0ca4db4533d06b8cf25d2114ea1.

## Data
Pediatric (ages 1-5) anterior-posterior chest X-rays from one hospital in Guangzhou. Labels were
graded by two physicians (a third checked the evaluation set). 5,856 unique images from 2,790
patient IDs (parsed from filenames): 1,583 NORMAL and 4,273 PNEUMONIA. The source description
states 5,863 images; the 7-image difference was not explained.

## Preparation steps
1. Inventory every image (path, label, size, mode, sha256); 0 corrupt files.
2. Exclude the identical nested copy and macOS metadata in the zip (reported, not deleted).
3. Parse patient IDs from filenames; build a patient-level 70/15/15 split (seed 42).
4. Verify with scripts/check_week2.py (splits.csv sha256, LF-normalised: 5cd9a952286fc4f40415c8fd78d2042f82755f31d74d6c4b90f6ea49d787a16f).

## Split
| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| train | 1,109 | 2,991 | 4,100 |
| val | 237 | 641 | 878 |
| test | 237 | 641 | 878 |

The original split was not used: its validation set has 16 images and 170 pneumonia patient IDs
appear in both train and test (320 of 390 test pneumonia images). Results are therefore not
directly comparable with published numbers on the original split.

## Limitations
- Single site, pediatric patients only; class imbalance of about 2.7:1.
- No clinical metadata to cross-check labels; patient IDs are inferred from filenames.
- Normal IDs from two filename series (IM- and NORMAL2-IM-) may collide; they were grouped together.
- Image size and colour mode correlate with the label (median 1654x1323 px NORMAL vs 1160x776 px
  PNEUMONIA; all 283 RGB images are PNEUMONIA). A height-only rule scores 84.8% against a 73.0%
  majority baseline, so models may use these artifacts.
- 32 redundant exact-duplicate files, all same-patient and same-split.
