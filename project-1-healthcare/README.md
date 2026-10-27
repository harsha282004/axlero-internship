# Project 1: Healthcare Image Classification

Diagnostic Lab Image Classification using Computer Vision
for the Axlero Data Science and Machine Learning Internship.

**Educational / research project. Not a diagnostic tool. Not clinically validated.
Results below are measured on one patient-level split of one public dataset and
must not be read as evidence of real-world or clinical performance.**

## 1. Overview

- **Problem:** binary classification of pediatric chest X-rays into `NORMAL` vs
  `PNEUMONIA`.
- **Dataset:** Kermany et al. (2018) chest X-ray pneumonia dataset, 5,856 images,
  2,790 patients. Full provenance, licence, known data-quality issues and the
  patient-level split are documented in [`data/README.md`](data/README.md) —
  read that file before touching the data pipeline.
- **Scope:** an internship research/learning exercise in building a reproducible
  training + evaluation + robustness-audit pipeline, not a product.
- **Model families implemented and trained** (`src/models.py`):
  1. `baseline` — a small from-scratch CNN.
  2. `densenet121` — torchvision DenseNet121, trained from scratch
     (`--pretrained` is supported but was not used for the final candidate).
- **Model configurations investigated for DenseNet121** (3 seeds each: 42, 43, 44):
  1. **orig** — standard aspect-preserving letterbox preprocessing, no
     training-only augmentation (`runs/densenet121_pad_seed{42,43,44}`).
  2. **aug** — orig + training-only randomised padding augmentation
     (`--pad-aug random`) (`runs/densenet121_padaug_seed{42,43,44}`).
  3. **edgeaug** — aug + training-only, independent randomised edge crop/mask
     augmentation (`--edge-aug random`) (`runs/densenet121_edgeaug_seed{42,43,44}`).
- **High-level workflow:** `data/splits.csv` (authoritative, patient-level split)
  → `src.train` (train + validation-only model/threshold selection) →
  `src.evaluate` (one split, usually the held-out test split) → optional
  robustness audits (`scripts/*audit*.py`, `scripts/*stress_test*.py`), all
  validation-split only.

## 2. Dataset Setup

The dataset is **not** stored in this repository. See
[`data/README.md`](data/README.md) for provenance, licence, known issues
(duplicate packaging, corrupt-file check, class-correlated image size/colour
mode, a shortcut-risk finding, etc.) and how `data/splits.csv` was built.

### Expected layout

Point `--data-root` (or the `CXR_DATA_ROOT` environment variable) at the folder
that **contains** the outer `chest_xray/` folder, e.g.:

```
C:\Users\<you>\dataset\kermany\
    chest_xray\
        train\NORMAL\...
        train\PNEUMONIA\...
        val\NORMAL\...
        val\PNEUMONIA\...
        test\NORMAL\...
        test\PNEUMONIA\...
```

`data/splits.csv` paths are relative to this root (e.g.
`chest_xray/train/NORMAL/IM-0001-0001.jpeg`) and work with either `/` or `\`
separators (`src/dataset.py:normalize_rel_path`).

### Manifest and splits

- `data/splits.csv` is the **only** authoritative source of labels and splits;
  the code never re-derives or edits splits from the raw folders.
- Columns: `path, label, subtype, patient_id, orig_split, split`.
- Patient-level 70/15/15 split, seed 42. `src/dataset.py:read_manifest` rejects
  the manifest outright if any patient appears in more than one split, if any
  path is duplicated, or if labels/splits are outside the expected set.

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| train | 1,109 | 2,991 | 4,100 |
| val   | 237   | 641   | 878   |
| test  | 237   | 641   | 878   |

### Validate your local copy before training

```
cd project-1-healthcare
.venv\Scripts\python.exe scripts\check_data.py <dataset_root>
.venv\Scripts\python.exe scripts\verify_splits.py
```
Both are read-only; `check_data.py` confirms every manifest path resolves under
your `<dataset_root>`, `verify_splits.py` re-confirms no patient-level leakage
and the `splits.csv` sha256 documented in `data/README.md`.

### Privacy / dataset handling

De-identified, publicly licensed (CC BY 4.0) pediatric chest X-rays; no
additional clinical metadata is present or used. The dataset and any derived
images/checkpoints must never be committed to this repository (see §8).

## 3. Environment Setup

- **Python:** 3.13 (verified).
- **Verified installed versions** (`pip show`/`__version__` at the time of the
  final evaluation in this document): `torch 2.11.0+cu128`,
  `torchvision 0.26.0+cu128`, `scikit-learn 1.9.1`, `numpy 2.5.2`,
  `pandas 3.0.6`, `pillow 12.3.0`, `matplotlib 3.11.2`.
  **Known inconsistency:** `requirements.txt` currently pins `pandas>=2.2,<3`,
  but the verified environment actually has `pandas 3.0.6` installed (outside
  that stated range) and everything in this document ran correctly against it.
  This is flagged, not silently fixed — tightening/loosening the pin is a
  dependency decision left for review, not changed here.
- **Install** (from `project-1-healthcare/`):
  ```
  python -m venv .venv
  .venv\Scripts\pip install -r requirements.txt
  ```
  `torch`/`torchvision` must be a matching pair; for a CUDA build install those
  two from the PyTorch index for your CUDA version first, then the rest of
  `requirements.txt` (see the comments in that file).
- **Verify the environment:**
  ```
  .venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```
- **CUDA:** training was run on an NVIDIA RTX 4050 Laptop GPU (6 GB VRAM),
  CUDA 12.8, with mixed precision (`torch.amp`) enabled by default on CUDA
  (`--no-amp` to disable).
- **CPU execution:** fully supported via `--device cpu` (used throughout
  `tests/test_pipeline.py`'s synthetic-data smoke tests); training on the real
  dataset on CPU will be substantially slower than the GPU timings below.

## 4. Training

Model selection and threshold selection use **validation data only**; the test
split is never read by `src.train`.

### Reproducing the final EdgeAug configuration (seeds 42/43/44)

Run from `project-1-healthcare/`, with your dataset root substituted:

```
.venv\Scripts\python.exe -m src.train --model densenet121 --preprocess pad --pad-aug random --edge-aug random --edge-aug-prob 0.3 --edge-grey-fill 128 --batch-size 16 --lr 1e-4 --epochs 20 --weight-decay 1e-4 --dropout 0.3 --seed 42 --data-root "<dataset_root>" --run-name densenet121_edgeaug_seed42
```
(repeat with `--seed 43`/`--run-name densenet121_edgeaug_seed43`, and
`--seed 44`/`--run-name densenet121_edgeaug_seed44`)

This is a **reproduction instruction**: these three runs already exist under
`runs/densenet121_edgeaug_seed{42,43,44}/` and were not rerun to produce this
document. Running the command again with the same `--run-name` against an
existing run will refuse to proceed (`src/train.py` raises `FileExistsError`
if `<run-dir>/best.pt` already exists) — use a different `--run-name` if you
want a fresh comparable run.

### What the command does

- **Model:** DenseNet121 (`torchvision.models.densenet121`), trained from
  scratch (`--pretrained` not passed), classifier head replaced with
  `Dropout(0.3) -> Linear(_, 1)`, one logit output (PNEUMONIA vs NORMAL).
- **Preprocessing** (`--preprocess pad`): `Grayscale(3)` (so single- and
  3-channel source JPEGs are normalised identically) → aspect-preserving
  resize (longer side → 224) → centred constant padding to 224x224 →
  `ToTensor` → fixed ImageNet `Normalize`. Deterministic at evaluation time.
- **Training-only augmentation** (both disabled by default; `edge_aug`
  requires no particular `--preprocess` mode, `pad_aug` requires
  `--preprocess pad`; both were enabled together for this configuration and
  are otherwise fully independent — see §6):
  - `--pad-aug random`: 50% of training images keep the standard centred
    black letterbox; otherwise the padding fill is randomised among black /
    image-mean / a random grey, and the resized content is placed at a random
    offset along the padded axis. Content is never cropped or distorted.
  - `--edge-aug random --edge-aug-prob 0.3 --edge-grey-fill 128`: with
    probability 0.3, a training image gets an outer-border crop or mask of
    width sampled uniformly from 0–10% of `min(width, height)`, filled with
    black / a fixed grey (128) / the image's own mean intensity (each choice
    equally likely). Operates on the original-resolution image, before the
    letterbox step, using the same original-image pixel-coordinate convention
    as the robustness audits (§6).
  - `RandomAffine`/`ColorJitter` (always on for training, unconditional on any
    flag) are applied afterwards, unchanged from the original pipeline.
- **Seed handling:** `--seed` seeds `python`/`numpy`/`torch`
  (`src/utils.py:set_seed`), sets `cudnn.deterministic=True` by default
  (`--no-deterministic` to opt out for speed), and seeds the DataLoader's
  generator + per-worker RNGs (`seed_worker`) so the training-data order and
  augmentation draws are reproducible for a fixed seed and worker count.
- **Optimizer/schedule:** AdamW, `lr=1e-4`, `weight_decay=1e-4`,
  `ReduceLROnPlateau` on validation loss (factor 0.5, patience 2).
- **Class imbalance:** `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` computed
  from the **training** split only (no leakage from val/test); disable with
  `--no-class-weights`.
- **Model selection:** best epoch = highest validation ROC-AUC
  (`--select-metric val_roc_auc`), early-stopped with `--patience 5`
  (0 disables early stopping), max `--epochs 20`.
- **Mixed precision:** `torch.amp` autocast + `GradScaler`, on by default when
  `device.type == "cuda"` (`--no-amp` to disable).
- **Threshold selection:** Youden's J statistic (`--threshold-strategy youden`)
  computed on validation predictions of the **best** checkpoint only, after
  training finishes.
- **Checkpoint/config/metadata output** (`runs/<run-name>/`, git-ignored):
  `best.pt` (model weights + `{name, dropout, preprocess, pad_aug, edge_aug}`
  metadata + `epoch`/`seed`/`select_metric`/`select_value`), `config.json`
  (every CLI argument, plus resolved `data_root`/`device`/`amp`/`pos_weight`),
  `history.csv`/`history.json` (per-epoch metrics), `threshold.json`
  (`{threshold, selected_on: "val", best_epoch, strategy, val_sensitivity,
  val_specificity, val_f1}`), `val_metrics.json` (full metric set at the
  selected threshold), `train.log`.
- **Dataset-root resolution:** `--data-root` or the `CXR_DATA_ROOT`
  environment variable; `src/utils.py:resolve_data_root` raises immediately if
  neither is set or the path isn't a directory.
- **Error handling:** fails loudly, never silently — missing images
  (`FileNotFoundError` with the first 10 missing paths listed), patient
  leakage or duplicate manifest paths (`ValueError` at manifest load), NaN/Inf
  training loss (`FloatingPointError`), an existing `run-name` (`FileExistsError`),
  no checkpoint ever saved because the selection metric was undefined every
  epoch (`RuntimeError`).

### Observed training time (this machine, RTX 4050, AMP on)

~38–65 s/epoch; the three `edgeaug` runs took 18, 20 and 20 epochs respectively
(seed 42 early-stopped) — roughly 12–14 minutes per seed, ~40 minutes total for
all three. (Measured from the existing `train.log` files; not rerun to produce
this document.)

## 5. Evaluation

### Validation-based model and threshold selection

Both happen automatically inside `src.train` (§4) — the best epoch by
validation ROC-AUC, then the decision threshold by Youden's J on that epoch's
validation predictions. The test split is never read during this process.

### How the frozen final evaluation was performed

A separate model-selection review (read-only, validation-and-robustness-audit
evidence only — see §6 and §7) selected **one** checkpoint for the single,
frozen look at test performance, run exactly once with `src.evaluate`:

```
cd project-1-healthcare
.venv\Scripts\python.exe -m src.evaluate --checkpoint runs\densenet121_edgeaug_seed43\best.pt --data-root "<dataset_root>" --manifest data\splits.csv --split test
```

- **Checkpoint:** `runs/densenet121_edgeaug_seed43/best.pt` (DenseNet121,
  `preprocess=pad`, `pad_aug=random`, `edge_aug=random`, seed 43, best epoch 19
  of a 20-epoch run).
- **Threshold used:** `0.5921655297279358` — the exact value from that run's
  `threshold.json` (`"selected_on": "val"`), **not** overridden (no
  `--threshold`/`--threshold-from-val` passed; `src.evaluate` defaults to the
  saved `threshold.json` value).
- **Output:** `runs/densenet121_edgeaug_seed43/eval_test/` (`metrics.json`,
  `predictions.csv`, `confusion_matrix.png`, `evaluate.log`) — this directory
  did not exist before this evaluation; `src.evaluate` now refuses to run if
  its output directory already exists and is non-empty (§6).
- This document is a **validation-only report** (§7 reports only validation
  metrics for all nine checkpoints). The frozen test-set numbers from this
  one-time evaluation exist in the artifacts listed above but are
  intentionally not reproduced, tabulated, or analysed here, so that
  model-family comparisons and conclusions in this document are not
  influenced, even indirectly, by test-split outcomes.

### Validation vs. test metrics — do not conflate

Every number under "Validation results" anywhere in this document (§7, and
the per-seed/per-family tables) comes from `val_metrics.json` /
`threshold.json`, produced during training on the **validation** split and
used for model/threshold selection. A separate, one-time `src.evaluate
--split test` run was performed on the single checkpoint selected above (see
`runs/densenet121_edgeaug_seed43/eval_test/`); its outputs exist on disk but
are deliberately kept out of this document's results and comparisons, which
are scoped to validation evidence only. Validation and test numbers are not
interchangeable, and §7 below reports validation metrics exclusively.

## 6. Robustness Analysis

All of the following are **validation-split-only, exploratory, synthetic**
perturbation studies — see "Limitations of synthetic perturbation results"
below before drawing any conclusion from them.

- **Padding augmentation** (`--pad-aug random`, `src/dataset.py:RandomPadLetterbox`):
  training-only; randomises the letterbox fill colour and offset without ever
  cropping or distorting the resized image; motivated by a finding that models
  trained with a fixed black letterbox could partly key on the fixed
  black-padding geometry rather than image content.
- **Edge augmentation** (`--edge-aug random`, `src/dataset.py:RandomEdgeMask`):
  training-only, independent of padding augmentation; randomly crops or masks
  an outer border (0–10% of the shorter side) with black/grey/image-mean
  fill, on the original-resolution image before the letterbox step; added to
  test and improve robustness to synthetic edge/margin perturbations while
  preserving the padding-augmentation robustness gain.
- **Synthetic perturbation audits** (`scripts/perturbation_audit_densenet.py`,
  `scripts/margin_sensitivity_audit.py`): apply hand-designed perturbations
  (grey/mean-filled padding bars, black/grey/mean-filled outer margins of
  5–10%, interior "ring" masks, centre crops) to validation images and
  re-score each of the nine DenseNet121 checkpoints (orig/aug/edgeaug ×
  3 seeds) with each checkpoint's own **fixed, saved** validation threshold
  (never retuned per condition). Every run verifies its "control" condition
  reproduces the saved validation predictions bit-for-bit before reporting
  anything, and cross-checks reused conditions against the earlier two-family
  audit (regression check).
- **Size-matched stress evaluation** (`scripts/size_matched_stress_test_densenet_all.py`):
  compares ROC-AUC within caliper-matched pairs and pixel-count/height bins so
  that an apparent "better" model isn't just better on a different mix of
  image sizes; uses the exact same 135 matched pairs (270 images) and bins for
  all three families, verified against an earlier six-run audit.

### Observed trade-offs (measured, validation-split only — see caveats below)

| Audit | orig | aug | edgeaug |
|---|---|---|---|
| Grey-padding (`pad_imgmean`) sensitivity | 0.650 (collapses) | 0.988 | 0.987 |
| Black-margin (`margin10_black`) specificity, delta vs. own control | −0.211 | −0.321 (worse) | **−0.145 (best)** |
| Image-mean-margin (`margin10_imgmean`) specificity, delta vs. own control | −0.008 (best) | −0.076 | −0.027 |
| Size-matched (135 pairs / 270 images) mean ROC-AUC | 0.9960 | 0.9949 | 0.9949 |

`edgeaug` preserves `aug`'s fix for the original model's grey-padding
shortcut-sensitivity and is additionally more robust to black-margin masking
than both other families — but it is **not uniformly better**: on
`margin10_imgmean` and `ring10_interior`, `edgeaug` shows a small,
statistically distinguishable (95% patient-bootstrap CI excludes 0)
specificity decline relative to `orig`, and its size-matched AUC is slightly
below `orig`'s. Full per-condition tables, bootstrap CIs, and alignment checks
are in `runs/perturbation_audit_edgeaug/`, `runs/margin_sensitivity_audit_edgeaug/`,
`runs/size_matched_stress_test_edgeaug/` (each with a `summary.json` and the
per-seed/per-checkpoint CSVs).

### Limitations of the synthetic perturbation results

- These perturbations are hand-designed and do not represent any confirmed
  real acquisition artefact — a change under a synthetic perturbation is **not
  proof of shortcut use or of real-world robustness**; masking can create
  out-of-distribution inputs or remove genuine anatomy, which can move metrics
  for reasons unrelated to shortcut reliance.
- Every audit's own `summary.json` is labelled `"Exploratory; validation only;
  inference only; test split never read"` — this is deliberate, not
  boilerplate, and should be preserved in any future reporting of these
  results.
- Three seeds per family is a small sample for estimating true seed-to-seed
  variance.
- None of this has been checked against the test split or any external
  dataset.

## 7. Results

### Validation results (all nine DenseNet121 checkpoints, mean ± sample SD over seeds 42/43/44)

| Family | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 | Accuracy |
|---|---|---|---|---|---|---|
| orig    | 0.9993 ± 0.0002 | 0.9997 ± 0.0001 | 0.9797 ± 0.0056 | 0.9972 ± 0.0049 | 0.9892 ± 0.0023 | 0.9844 ± 0.0033 |
| aug     | 0.9990 ± 0.0001 | 0.9997 ± 0.0001 | 0.9870 ± 0.0032 | 0.9873 ± 0.0042 | 0.9911 ± 0.0016 | 0.9871 ± 0.0024 |
| edgeaug | 0.9989 ± 0.0002 | 0.9996 ± 0.0001 | 0.9860 ± 0.0027 | 0.9902 ± 0.0088 | 0.9911 ± 0.0004 | 0.9871 ± 0.0007 |

Per-seed validation metrics: `runs/<run-name>/val_metrics.json` and
`threshold.json` for each of the nine runs; a candidate-selection writeup with
the full comparability checks is preserved in this session's record (not
duplicated verbatim here to avoid two sources of truth — the table above is
the authoritative summary, recomputed directly from those nine JSON files).

This report is scoped to validation evidence: model-family comparisons,
the robustness-audit discussion in §6, and the table above all use
validation-split metrics only. A one-time frozen test-set evaluation of the
selected checkpoint (`densenet121_edgeaug_seed43`) was performed following
the procedure in §5 and its artifacts exist under
`runs/densenet121_edgeaug_seed43/eval_test/`; those numbers are intentionally
not reproduced, tabulated, or discussed in this document.

## 8. Limitations and Future Work

- **Single dataset, single site, single population:** pediatric (ages 1–5),
  one hospital (Guangzhou Women and Children's Medical Center), one
  acquisition pipeline. No claim is made, or should be inferred, about
  performance on adults, other hospitals, other scanners, or other
  populations.
- **Limited seeds:** three seeds per configuration (42/43/44); seed-to-seed
  variability estimates (±SD above) are themselves noisy with n=3.
- **Synthetic robustness audits only:** see §6's limitations — exploratory,
  hypothesis-generating, not proof of real-world robustness.
- **Possible dataset-specific shortcuts:** `data/README.md` documents that
  image size and colour mode correlate with label in this dataset and that a
  single height threshold alone classifies 84.8% of images correctly — models
  trained on this data may be partially exploiting such artefacts rather than
  anatomy; the perturbation/size-matched audits were designed to probe this,
  not to rule it out.
- **No independent external validation:** only this dataset's own
  train/val/test split was used; no cross-dataset or prospective evaluation
  was performed.
- **One frozen test evaluation:** by design (§5), only one checkpoint was
  evaluated on test, exactly once, with no tuning in response to the result.
  This protects validity but means there is no test-split comparison across
  the three families — such a comparison, if wanted, would need to be
  explicitly planned (ideally pre-registered) as separate future work, not
  retrofitted onto this result.
- **`pandas` version pin mismatch** (§3) — flagged, not resolved here.
- **Not clinically validated, not a diagnostic tool, not for clinical or
  production use.**

## 9. Repository Hygiene

- **Git-ignored** (see `.gitignore`): `runs/` (all training outputs, audit
  outputs, checkpoints, evaluation results), the raw dataset (`chest_xray/`),
  virtual environments (`.venv/`), Python/`pytest` caches, model weight file
  extensions (`*.pt`, `*.pth`, `*.ckpt`, etc.).
- **Intentionally tracked:** `data/splits.csv`, `data/README.md`,
  `data/eda/*.png`, `data/*.csv` (manifest/inventory files — small, no raw
  images), `notebooks/*.ipynb`.
- **Where results actually live locally:** everything under
  `project-1-healthcare/runs/<run-name>/` for training runs, and
  `project-1-healthcare/runs/<audit-name>/` for robustness audits and the
  frozen test evaluation (`runs/densenet121_edgeaug_seed43/eval_test/`). None
  of this is, or should be, committed.
- **Commands above all run from `project-1-healthcare/`** using
  `.venv\Scripts\python.exe` (or `python -m ...` with the venv activated), as
  documented in each section.
