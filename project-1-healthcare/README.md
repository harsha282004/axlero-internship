# Project 1: Healthcare Image Classification

## Overview

This project explores binary classification of chest X-ray images into two classes: `NORMAL` and `PNEUMONIA`, using deep learning.

We use a pretrained ResNet50 convolutional neural network (CNN) for baseline training and fine-tuning.

**Disclaimer:** This is an educational machine learning project, not a clinically validated diagnostic system.

## Project Structure

```text
project-1-healthcare/
├── data/
│   ├── splits.csv
│   └── README.md
├── src/
│   └── modeling/
│       ├── train.py
│       └── finetune.py
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── plots/
├── requirements.txt
└── README.md
```

## Dataset

The current dataset split contains 5,856 images.

| Split      | Images |
| ---------- | -----: |
| Training   |  4,100 |
| Validation |    878 |
| Test       |    878 |
| Total      |  5,856 |

The scripts use `data/splits.csv` to determine each image's split. The CSV should contain `path`, `label`, and `split` columns.

The original dataset source and attribution information are documented in `data/README.md`.

Patient-level separation must be verified against the original patient metadata before claiming that there is no patient overlap between splits.

## Model Configuration

### Baseline ResNet50

* Pretrained ResNet50 architecture.
* Input image size: 224 × 224.
* Batch size: 8.
* Optimizer: Adam.
* Learning rate: 0.0001.
* Training duration: 2 epochs.
* The final classification layer is trained while the backbone remains frozen.
* The best checkpoint is selected using validation loss.

### Fine-Tuned ResNet50

* Initialized from the best baseline checkpoint.
* The `layer4` block and classification layer are trainable.
* `layer4` learning rate: 0.00001.
* Classification layer learning rate: 0.00005.
* Optimizer: Adam.
* Batch size: 8.
* Training duration: 2 epochs.
* The best checkpoint is selected using validation loss.

## Environment Setup

Use Python 3.11 and a compatible PyTorch/torchvision installation.

Create a Conda environment:

```powershell
conda create -n axlero-ml python=3.11 -y
conda activate axlero-ml
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If PyTorch is not installed, follow the official installation instructions to select compatible PyTorch and torchvision versions for your hardware.

## Reproducing the Experiments

Run commands from the repository root.

First, train the baseline:

```powershell
python project-1-healthcare/src/modeling/train.py
```

Then fine-tune the model:

```powershell
python project-1-healthcare/src/modeling/finetune.py
```

Fine-tuning requires the baseline checkpoint produced by the first script.

The scripts may download pretrained ImageNet weights if they are not already cached. An internet connection may be required.

Check each script's configuration and output paths before running it. Training may take considerable time on a CPU.

## Recorded Results

The following are the currently recorded experimental results. They must be checked against the generated metric files when reproducing the experiments.

| Metric          | Baseline ResNet50 | Fine-Tuned ResNet50 |
| --------------- | ----------------: | ------------------: |
| Test accuracy   |            93.51% |              96.81% |
| Macro precision |            94.03% |              96.07% |
| Macro recall    |            89.30% |              95.82% |
| Macro F1-score  |            91.32% |              95.94% |

Detailed metrics and plots are available in `outputs/metrics/` and `outputs/plots/`.

## Evaluation Methodology

The training split is used for optimization, and the validation split is used for checkpoint selection. The selected checkpoint is evaluated on the test split.

The evaluation includes accuracy, class-specific precision, recall and F1-score, and a confusion matrix.

The test set should not be used to select hyperparameters or checkpoints.

## Limitations

* Each experiment was run for only two epochs.
* The classes are imbalanced, with more PNEUMONIA images than NORMAL images.
* Patient-level independence has not been established by the CSV split counts alone.
* Results may vary depending on the software environment, hardware and random operations.
* Generalization to other hospitals and patient populations has not been established.
* These experimental results must not be used for clinical diagnosis or medical decisions.
