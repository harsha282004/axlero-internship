from pathlib import Path
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "splits.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
SEED = 42

# Index 0 = NORMAL, index 1 = PNEUMONIA
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# 3. Dataset
# --------------------------------------------------

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = PROJECT_ROOT / str(row["path"]).replace("/", "\\")

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

        label_name = str(row["label"]).strip().upper()

        if label_name not in CLASS_TO_INDEX:
            raise ValueError(f"Unexpected label: {label_name}")

        label = CLASS_TO_INDEX[label_name]

        return image, label


# --------------------------------------------------
# 4. Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

evaluation_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# 5. Load and validate the CSV
# --------------------------------------------------

def load_splits():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    dataframe = pd.read_csv(CSV_PATH)

    required_columns = {"path", "label", "split"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing CSV columns: {missing_columns}")

    dataframe["split"] = dataframe["split"].astype(str).str.strip().str.lower()
    dataframe["label"] = dataframe["label"].astype(str).str.strip().str.upper()

    if not set(dataframe["label"].unique()).issubset(CLASS_TO_INDEX):
        raise ValueError(
            f"Unexpected labels: {dataframe['label'].unique().tolist()}"
        )

    expected_splits = {"train", "val", "test"}

    if set(dataframe["split"].unique()) != expected_splits:
        raise ValueError(
            f"Expected splits {expected_splits}, "
            f"found {set(dataframe['split'].unique())}"
        )

    # Fail early if any image is missing.
    missing_paths = []

    for relative_path in dataframe["path"]:
        path = PROJECT_ROOT / str(relative_path).replace("/", "\\")

        if not path.is_file():
            missing_paths.append(str(path))

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} images are missing. "
            f"First missing path: {missing_paths[0]}"
        )

    return dataframe


# --------------------------------------------------
# 6. Build ResNet50
# --------------------------------------------------

def build_model():
    # Use official pretrained ImageNet weights.
    # The first run may download weights from the internet.
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(CLASS_NAMES))

    return model


# --------------------------------------------------
# 7. Training and validation
# --------------------------------------------------

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)

    return average_loss, accuracy


# --------------------------------------------------
# 8. Final evaluation
# --------------------------------------------------

def evaluate_model(model, loader):
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            outputs = model(images.to(DEVICE))
            predictions = outputs.argmax(dim=1).cpu().tolist()

            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=[0, 1],
        zero_division=0,
    )

    matrix = confusion_matrix(all_labels, all_predictions, labels=[0, 1])

    results = {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "normal_precision": float(precision[0]),
        "normal_recall": float(recall[0]),
        "normal_f1": float(f1[0]),
        "pneumonia_precision": float(precision[1]),
        "pneumonia_recall": float(recall[1]),
        "pneumonia_f1": float(f1[1]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }

    print("\nFinal test results")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    return results


# --------------------------------------------------
# 9. Save training curves
# --------------------------------------------------

def save_training_plot(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet50 Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "loss_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    plt.plot(epochs, history["val_accuracy"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ResNet50 Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "accuracy_curves.png", dpi=150)
    plt.close()


# --------------------------------------------------
# 10. Main training pipeline
# --------------------------------------------------

def main():
    set_seed()

    for directory in [CHECKPOINT_DIR, PLOT_DIR, METRICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Project: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"CPU-only training: {DEVICE.type == 'cpu'}")

    dataframe = load_splits()

    train_df = dataframe[dataframe["split"] == "train"].copy()
    val_df = dataframe[dataframe["split"] == "val"].copy()
    test_df = dataframe[dataframe["split"] == "test"].copy()

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    print("\nTraining class counts:")
    print(train_df["label"].value_counts())

    train_dataset = ChestXrayDataset(train_df, train_transform)
    val_dataset = ChestXrayDataset(val_df, evaluation_transform)
    test_dataset = ChestXrayDataset(test_df, evaluation_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\nLoading pretrained ResNet50...")
    model = build_model().to(DEVICE)

    # Baseline: train the final classification layer first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    checkpoint_path = CHECKPOINT_DIR / "resnet50_baseline_best.pth"

    for epoch in range(EPOCHS):
        start_time = time.time()

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer
        )

        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        elapsed = time.time() - start_time

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val accuracy: {val_accuracy:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "best_val_loss": best_val_loss,
                    "seed": SEED,
                },
                checkpoint_path,
            )

            print(f"Saved best checkpoint: {checkpoint_path}")

    with open(METRICS_DIR / "training_history.json", "w") as file:
        json.dump(history, file, indent=2)

    save_training_plot(history)

    # Restore the best validation checkpoint before test evaluation.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_model(model, test_loader)

    results["best_epoch"] = checkpoint["epoch"]
    results["best_validation_loss"] = checkpoint["best_val_loss"]
    results["test_samples"] = len(test_dataset)

    with open(METRICS_DIR / "test_metrics.json", "w") as file:
        json.dump(results, file, indent=2)

    print("\nTraining complete.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {METRICS_DIR / 'test_metrics.json'}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
