"""Manifest loading and PyTorch Dataset/DataLoader for the chest X-ray images.

data/splits.csv is the authoritative split manifest. This module never creates or
changes splits; it only validates and reads them.
"""
import logging
from pathlib import Path, PurePosixPath

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.utils import LABEL_TO_INDEX, SPLITS, seed_worker

log = logging.getLogger("cxr")

IMAGE_SIZE = 224
# Fixed ImageNet statistics (constants, not fitted on any split, so no leakage).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = ["path", "label", "split", "patient_id"]
MAX_LISTED = 10


PREPROCESS_MODES = ("resize", "pad")


class LetterboxSquare:
    """Aspect-preserving resize (longer side -> size) then centred constant padding to size x size.

    Deterministic. Note: the amount of padding still reflects the original aspect ratio.
    """

    def __init__(self, size=IMAGE_SIZE, fill=0):
        self.size, self.fill = size, fill

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = TF.resize(img, [nh, nw])  # same interpolation/antialiasing as transforms.Resize
        canvas = Image.new(img.mode, (self.size, self.size), (self.fill,) * len(img.getbands()))
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


def build_transforms(train, preprocess="resize"):
    """Training: light augmentation. Eval: deterministic resize + normalise only.

    preprocess: "resize" (default, squash to 224x224) or "pad" (aspect-preserving + padding).
    """
    if preprocess not in PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocess {preprocess!r}; choose from {PREPROCESS_MODES}")
    fit = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)) if preprocess == "resize" else LetterboxSquare()
    head = [transforms.Grayscale(num_output_channels=3),  # also normalises the RGB files
            fit]
    tail = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    if not train:
        return transforms.Compose(head + tail)
    augment = [transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
               transforms.ColorJitter(brightness=0.2, contrast=0.2)]
    return transforms.Compose(head + augment + tail)


def normalize_rel_path(p):
    """Accept Windows or POSIX separators; reject absolute paths and '..'."""
    posix = str(p).strip().replace("\\", "/")
    pp = PurePosixPath(posix)
    if pp.is_absolute() or ".." in pp.parts or ":" in posix:
        raise ValueError(f"Manifest path must be relative and inside the dataset root: {p!r}")
    return posix


def read_manifest(manifest_path):
    """Read and validate the whole manifest (labels, splits, duplicates, patient leakage)."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest {manifest_path} lacks columns {missing}; has {list(df.columns)}")

    bad_labels = sorted(set(df.label) - set(LABEL_TO_INDEX))
    if bad_labels:
        raise ValueError(f"Invalid labels in manifest (expected {list(LABEL_TO_INDEX)}): {bad_labels}")
    bad_splits = sorted(set(df.split) - set(SPLITS))
    if bad_splits:
        raise ValueError(f"Invalid split names in manifest (expected {list(SPLITS)}): {bad_splits}")
    if (df.patient_id.str.strip() == "").any():
        raise ValueError("Manifest has empty patient_id values; cannot guarantee patient-level splits")

    df["rel_path"] = df.path.map(normalize_rel_path)
    dup = df.rel_path[df.rel_path.duplicated()].unique()
    if len(dup):
        raise ValueError(f"{len(dup)} duplicate paths in manifest, e.g. {list(dup[:MAX_LISTED])}")

    n_splits = df.groupby("patient_id").split.nunique()
    leaked = n_splits[n_splits > 1].index.tolist()
    if leaked:
        raise ValueError(f"Patient leakage: {len(leaked)} patient_id values appear in more than one "
                         f"split, e.g. {leaked[:MAX_LISTED]}")

    df["target"] = df.label.map(LABEL_TO_INDEX).astype(int)
    return df


def load_split(manifest_path, data_root, split):
    """Rows of one split with absolute paths; raises if any image file is missing."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    df = read_manifest(manifest_path)
    sub = df[df.split == split].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"Split {split!r} has no rows in {manifest_path}")
    root = Path(data_root)
    sub["abs_path"] = [str(root.joinpath(*p.split("/"))) for p in sub.rel_path]
    missing = [p for p in sub.abs_path if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(sub)} images for split {split!r} not found under {root}. "
            f"First missing: {missing[:MAX_LISTED]}. The dataset root must be the folder that "
            "contains chest_xray/ (see scripts/check_data.py).")
    return sub


class ChestXrayDataset(Dataset):
    """Returns (image tensor [3,224,224], label long)."""

    def __init__(self, frame, transform):
        self.paths = frame.abs_path.tolist()
        self.targets = frame.target.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        try:
            with Image.open(path) as im:
                im.load()
                x = self.transform(im)
        except Exception as e:  # never skip silently: surface the offending file
            raise RuntimeError(f"Failed to read/transform image {path}: {e}") from e
        return x, torch.tensor(self.targets[i], dtype=torch.long)


def make_loader(manifest_path, data_root, split, batch_size, num_workers=2, seed=42,
                train=None, pin_memory=False, preprocess="resize"):
    """DataLoader for one split. `train` defaults to (split == 'train') and controls
    shuffling and augmentation. Returns (loader, frame)."""
    train = (split == "train") if train is None else train
    frame = load_split(manifest_path, data_root, split)
    ds = ChestXrayDataset(frame, build_transforms(train, preprocess))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                        worker_init_fn=seed_worker, generator=gen, pin_memory=pin_memory,
                        persistent_workers=num_workers > 0, drop_last=False)
    return loader, frame


def class_counts(frame):
    """(n_normal, n_pneumonia) for a frame."""
    counts = frame.target.value_counts()
    return int(counts.get(0, 0)), int(counts.get(1, 0))
