from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOT_DIR = OUTPUT_DIR / "plots"

BASELINE_CHECKPOINT = (
    CHECKPOINT_DIR / "resnet50_baseline_best.pth"
)

FINETUNED_CHECKPOINT = (
    CHECKPOINT_DIR / "resnet50_finetuned_best.pth"
)

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2
SEED = 42

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 5e-5

NUM_WORKERS = 0

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. DATASET
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = (
            PROJECT_ROOT
            / str(row["path"]).replace("/", "\\")
        )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        label_name = str(row["label"]).strip().upper()

        if label_name not in CLASS_TO_INDEX:
            raise ValueError(
                f"Unexpected label: {label_name}"
            )

        label = CLASS_TO_INDEX[label_name]

        return image, label


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

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


# ============================================================
# 5. LOAD AND VALIDATE DATA
# ============================================================

def load_splits():

    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"Splits CSV not found: {CSV_PATH}"
        )

    dataframe = pd.read_csv(CSV_PATH)

    required_columns = {"path", "label", "split"}

    missing_columns = required_columns - set(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing CSV columns: {missing_columns}"
        )

    dataframe["split"] = (
        dataframe["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    expected_splits = {"train", "val", "test"}

    if set(dataframe["split"].unique()) != expected_splits:
        raise ValueError(
            f"Unexpected splits: "
            f"{dataframe['split'].unique().tolist()}"
        )

    if not set(dataframe["label"].unique()).issubset(
        CLASS_TO_INDEX
    ):
        raise ValueError(
            f"Unexpected labels: "
            f"{dataframe['label'].unique().tolist()}"
        )

    if dataframe["path"].duplicated().any():
        raise ValueError(
            "Duplicate image paths found in splits.csv."
        )

    # Verify image paths.
    for relative_path in dataframe["path"]:

        image_path = (
            PROJECT_ROOT
            / str(relative_path).replace("/", "\\")
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

    return dataframe


# ============================================================
# 6. BUILD MODEL AND LOAD BASELINE CHECKPOINT
# ============================================================

def build_model():

    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Baseline checkpoint not found: "
            f"{BASELINE_CHECKPOINT}"
        )

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES),
    )

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Loaded baseline checkpoint:")
    print(BASELINE_CHECKPOINT)

    # Freeze all layers initially.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Unfreeze the final residual block.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    # Unfreeze the classifier.
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: {trainable_parameters:,}"
    )

    return model


# ============================================================
# 7. TRAINING AND VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    description="Training",
):

    is_training = optimizer is not None

    model.train(is_training)

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.set_grad_enabled(is_training):

        for images, labels in tqdm(
            loader,
            desc=description,
            leave=False,
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)

            total_loss += (
                loss.item() * images.size(0)
            )

            all_labels.extend(
                labels.detach().cpu().tolist()
            )

            all_predictions.extend(
                predictions.detach().cpu().tolist()
            )

    average_loss = (
        total_loss / len(loader.dataset)
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=[0, 1],
            zero_division=0,
        )
    )

    return {
        "loss": float(average_loss),
        "accuracy": float(
            accuracy_score(
                all_labels,
                all_predictions,
            )
        ),
        "normal_recall": float(recall[0]),
        "pneumonia_recall": float(recall[1]),
        "macro_f1": float(np.mean(f1)),
    }


# ============================================================
# 8. EVALUATE ACTUAL MODEL PREDICTIONS
# ============================================================

def evaluate_test_set(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in tqdm(
            loader,
            desc="Final test evaluation",
        ):

            outputs = model(images.to(DEVICE))

            predictions = (
                outputs.argmax(dim=1)
                .cpu()
                .tolist()
            )

            all_predictions.extend(predictions)

            all_labels.extend(
                labels.tolist()
            )

    # The report uses actual predictions from the model.
    report_text = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=[0, 1],
            zero_division=0,
        )
    )

    results = {
        "accuracy": float(
            accuracy_score(
                all_labels,
                all_predictions,
            )
        ),
        "normal_precision": float(precision[0]),
        "normal_recall": float(recall[0]),
        "normal_f1": float(f1[0]),
        "pneumonia_precision": float(precision[1]),
        "pneumonia_recall": float(recall[1]),
        "pneumonia_f1": float(f1[1]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report_dict,
    }

    print("\nActual test classification report:")
    print(report_text)

    print(
        "Confusion matrix "
        "(rows=true, columns=predicted):"
    )
    print(matrix)

    return results


# ============================================================
# 9. PLOT TRAINING HISTORY
# ============================================================

def save_plots(history):

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    # Loss curves.
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet50 Fine-tuning Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "finetune_loss_curves.png",
        dpi=150,
    )

    plt.close()

    # Validation recall curves.
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["val_normal_recall"],
        marker="o",
        label="NORMAL recall",
    )

    plt.plot(
        epochs,
        history["val_pneumonia_recall"],
        marker="o",
        label="PNEUMONIA recall",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Validation Recall During Fine-tuning")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "finetune_validation_recall.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():

    set_seed()

    for directory in [
        CHECKPOINT_DIR,
        METRICS_DIR,
        PLOT_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("Project:", PROJECT_ROOT)
    print("Device:", DEVICE)

    dataframe = load_splits()

    train_df = dataframe[
        dataframe["split"] == "train"
    ].copy()

    val_df = dataframe[
        dataframe["split"] == "val"
    ].copy()

    test_df = dataframe[
        dataframe["split"] == "test"
    ].copy()

    print("\nSplit sizes:")
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))

    train_dataset = ChestXrayDataset(
        train_df,
        train_transform,
    )

    val_dataset = ChestXrayDataset(
        val_df,
        evaluation_transform,
    )

    test_dataset = ChestXrayDataset(
        test_df,
        evaluation_transform,
    )

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

    model = build_model().to(DEVICE)

    optimizer = torch.optim.Adam([
        {
            "params": model.layer4.parameters(),
            "lr": BACKBONE_LR,
        },
        {
            "params": model.fc.parameters(),
            "lr": CLASSIFIER_LR,
        },
    ])

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "val_normal_recall": [],
        "val_pneumonia_recall": [],
        "val_macro_f1": [],
    }

    best_val_loss = float("inf")
    best_epoch = None
    best_state = None

    for epoch in range(EPOCHS):

        start_time = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            description="Training",
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            description="Validation",
        )

        history["train_loss"].append(
            train_metrics["loss"]
        )

        history["val_loss"].append(
            val_metrics["loss"]
        )

        history["train_accuracy"].append(
            train_metrics["accuracy"]
        )

        history["val_accuracy"].append(
            val_metrics["accuracy"]
        )

        history["val_normal_recall"].append(
            val_metrics["normal_recall"]
        )

        history["val_pneumonia_recall"].append(
            val_metrics["pneumonia_recall"]
        )

        history["val_macro_f1"].append(
            val_metrics["macro_f1"]
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}\n"
            f"Training loss: {train_metrics['loss']:.4f}\n"
            f"Validation loss: {val_metrics['loss']:.4f}\n"
            f"Validation accuracy: "
            f"{val_metrics['accuracy']:.4f}\n"
            f"NORMAL recall: "
            f"{val_metrics['normal_recall']:.4f}\n"
            f"PNEUMONIA recall: "
            f"{val_metrics['pneumonia_recall']:.4f}\n"
            f"Validation macro F1: "
            f"{val_metrics['macro_f1']:.4f}\n"
            f"Elapsed time: "
            f"{time.time() - start_time:.1f}s"
        )

        # Select the best checkpoint using validation loss.
        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            best_epoch = epoch + 1

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "best_val_loss": best_val_loss,
                    "seed": SEED,
                },
                FINETUNED_CHECKPOINT,
            )

            print(
                "Saved best checkpoint:",
                FINETUNED_CHECKPOINT,
            )

    with open(
        METRICS_DIR / "finetune_history.json",
        "w",
    ) as file:
        json.dump(history, file, indent=2)

    save_plots(history)

    if best_state is None:
        raise RuntimeError(
            "No valid fine-tuned checkpoint was saved."
        )

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    print(
        f"\nBest validation epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: {best_val_loss:.4f}"
    )

    # Evaluate the test set after model selection.
    test_results = evaluate_test_set(
        model,
        test_loader,
    )

    test_results["best_epoch"] = best_epoch
    test_results["best_validation_loss"] = (
        best_val_loss
    )
    test_results["test_samples"] = len(test_dataset)

    metrics_path = (
        METRICS_DIR / "finetune_test_metrics.json"
    )

    with open(metrics_path, "w") as file:
        json.dump(
            test_results,
            file,
            indent=2,
        )

    print("\nFine-tuning complete.")
    print("Checkpoint:", FINETUNED_CHECKPOINT)
    print(
        "History:",
        METRICS_DIR / "finetune_history.json",
    )
    print("Test metrics:", metrics_path)
    print("Plots:", PLOT_DIR)


if __name__ == "__main__":
    main()
