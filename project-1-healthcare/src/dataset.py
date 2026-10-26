"""Manifest loading and PyTorch Dataset/DataLoader for the chest X-ray images.

data/splits.csv is the authoritative split manifest. This module never creates or
changes splits; it only validates and reads them.
"""
import logging
from pathlib import Path, PurePosixPath

import numpy as np
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


PAD_AUG_MODES = ("none", "random")
PAD_AUG_KEEP_STANDARD_PROB = 0.5
PAD_AUG_GREY_RANGE = (64, 192)

EDGE_AUG_MODES = ("none", "random")
EDGE_AUG_PROB = 0.3
EDGE_AUG_MAX_FRAC = 0.10
EDGE_AUG_GREY_FILL = 128
EDGE_AUG_OPS = ("crop", "mask")
EDGE_AUG_FILL_CHOICES = ("black0", "grey_fixed", "image_mean")


class RandomPadLetterbox:
    """TRAINING-ONLY randomised letterbox padding (same resize as LetterboxSquare, content never altered).

    With probability keep_standard_prob the standard centred black letterbox is used. Otherwise:
      fill   : one of {0 (black), mean grayscale intensity of the original image, random grey} with equal
               probability; random grey is an integer drawn uniformly from grey_range (inclusive)
      offset : the resized image is placed at a uniformly random position along the padded axis
               (anywhere from flush-start to flush-end); the image is never cropped or distorted.
    Randomness comes from the torch RNG, which the DataLoader seeds per worker (reproducible for a fixed
    seed and number of workers). Expects a 3-channel grayscale PIL image (after Grayscale(3)).
    """

    def __init__(self, size=IMAGE_SIZE, keep_standard_prob=PAD_AUG_KEEP_STANDARD_PROB,
                 grey_range=PAD_AUG_GREY_RANGE):
        lo, hi = grey_range
        if not (0 <= lo <= hi <= 255):
            raise ValueError(f"grey_range must satisfy 0 <= lo <= hi <= 255, got {grey_range}")
        if not (0.0 <= keep_standard_prob <= 1.0):
            raise ValueError("keep_standard_prob must be in [0, 1]")
        self.size, self.keep_standard_prob, self.grey_range = size, keep_standard_prob, (int(lo), int(hi))

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        standard = torch.rand(1).item() < self.keep_standard_prob
        fill, ox, oy = 0, (self.size - nw) // 2, (self.size - nh) // 2
        if not standard:
            choice = int(torch.randint(0, 3, (1,)))
            if choice == 1:
                fill = int(round(float(np.asarray(img)[..., 0].mean())))
            elif choice == 2:
                fill = int(torch.randint(self.grey_range[0], self.grey_range[1] + 1, (1,)))
            ox = int(torch.randint(0, self.size - nw + 1, (1,)))
            oy = int(torch.randint(0, self.size - nh + 1, (1,)))
        img = TF.resize(img, [nh, nw])
        canvas = Image.new(img.mode, (self.size, self.size), (fill,) * len(img.getbands()))
        canvas.paste(img, (ox, oy))
        return canvas


class RandomEdgeMask:
    """TRAINING-ONLY randomised edge crop/mask, applied to the ORIGINAL-resolution image BEFORE the fit
    (resize/letterbox) step. Independent of RandomPadLetterbox: this perturbs the image content itself,
    pad_aug perturbs only the letterbox fill/offset; both may be enabled together.

    Same original-image pixel-coordinate convention as scripts/margin_sensitivity_audit.py and
    scripts/perturbation_audit_densenet.py: for an image of size w x h, width m = round(frac * min(w, h)),
    frac drawn uniformly from [0, max_frac).

    With probability (1 - prob) the image is returned unchanged (no RNG state consumed beyond that one
    draw). Otherwise, with equal probability:
      crop : img.crop((m, m, w-m, h-m)) -- removes an m-pixel border on every side (identical to the
             audits' "trim" condition). The image is simply smaller; the downstream fit step resizes it
             to the canvas exactly as it would any other image size, so content is uniformly magnified,
             never stretched asymmetrically, and no separate resize logic is needed here.
      mask : an m-pixel border band (x < m or x >= w-m or y < m or y >= h-m) is set to `fill` on every
             channel (identical to the audits' "margin" condition). Image dimensions are unchanged; only
             pixels in the border band are altered, so the interior anatomy is never touched.
    `fill` is drawn uniformly among {0 (black), grey_fill (fixed grey), round(mean of THIS image)}, with
    the same value written to all channels (the input is already 3-channel-identical grayscale).
    m is capped by max_frac <= 0.10 (so 2*m < min(w, h)) and rounds to 0 for degenerate tiny images,
    in which case this is a no-op; crop/mask therefore can never create a zero or negative dimension.
    Randomness comes from the torch RNG (same seeding story as RandomPadLetterbox).
    """

    def __init__(self, prob=EDGE_AUG_PROB, max_frac=EDGE_AUG_MAX_FRAC, grey_fill=EDGE_AUG_GREY_FILL):
        if not (0.0 <= prob <= 1.0):
            raise ValueError("prob must be in [0, 1]")
        if not (0.0 <= max_frac <= 0.10):
            raise ValueError("max_frac must be in [0, 0.10]")
        if not (0 <= grey_fill <= 255):
            raise ValueError("grey_fill must be in [0, 255]")
        self.prob, self.max_frac, self.grey_fill = prob, max_frac, int(grey_fill)

    def __call__(self, img):
        if torch.rand(1).item() >= self.prob:
            return img
        w, h = img.size
        frac = torch.rand(1).item() * self.max_frac
        m = round(frac * min(w, h))
        if m <= 0:
            return img
        op = EDGE_AUG_OPS[int(torch.randint(0, 2, (1,)))]
        choice = int(torch.randint(0, 3, (1,)))
        if choice == 0:
            fill = 0
        elif choice == 1:
            fill = self.grey_fill
        else:
            fill = int(round(float(np.asarray(img)[..., 0].mean())))
        return self.apply(img, m, op, fill)

    @staticmethod
    def apply(img, m, op, fill):
        """Pure crop/mask operation (no randomness): exposed so it can be tested directly with an
        exact, known (m, op, fill) instead of having to reverse-engineer them from pixel differences."""
        w, h = img.size
        if m <= 0:
            return img
        if op == "crop":
            return img.crop((m, m, w - m, h - m))
        if op == "mask":
            a = np.array(img)
            a[:m], a[h - m:], a[:, :m], a[:, w - m:] = fill, fill, fill, fill
            return Image.fromarray(a, img.mode)
        raise ValueError(f"Unknown op {op!r}; choose from {EDGE_AUG_OPS}")


def build_transforms(train, preprocess="resize", pad_aug="none", pad_grey_range=PAD_AUG_GREY_RANGE,
                     edge_aug="none", edge_aug_prob=EDGE_AUG_PROB, edge_max_frac=EDGE_AUG_MAX_FRAC,
                     edge_grey_fill=EDGE_AUG_GREY_FILL):
    """Training: light augmentation. Eval: deterministic resize + normalise only.

    preprocess: "resize" (default, squash to 224x224) or "pad" (aspect-preserving + padding).
    pad_aug: "none" (default; identical to the original pipeline) or "random" (RandomPadLetterbox, TRAINING
    only, requires preprocess="pad"). It is ignored when train=False so evaluation never changes.
    edge_aug: "none" (default; identical to the pipeline without this option) or "random"
    (RandomEdgeMask, TRAINING only, independent of pad_aug and of preprocess). It is ignored when
    train=False so evaluation never changes. When "none", no RandomEdgeMask instance is added to the
    pipeline at all (not even a no-op one), so the RNG draw sequence -- and therefore every existing
    seeded/reproducible behaviour -- is completely unaffected by this option's presence.
    """
    if preprocess not in PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocess {preprocess!r}; choose from {PREPROCESS_MODES}")
    if pad_aug not in PAD_AUG_MODES:
        raise ValueError(f"Unknown pad_aug {pad_aug!r}; choose from {PAD_AUG_MODES}")
    if pad_aug == "random" and preprocess != "pad":
        raise ValueError("pad_aug='random' requires preprocess='pad'")
    if edge_aug not in EDGE_AUG_MODES:
        raise ValueError(f"Unknown edge_aug {edge_aug!r}; choose from {EDGE_AUG_MODES}")
    fit = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)) if preprocess == "resize" else LetterboxSquare()
    if train and pad_aug == "random":
        fit = RandomPadLetterbox(grey_range=pad_grey_range)
    head = [transforms.Grayscale(num_output_channels=3)]  # also normalises the RGB files
    if train and edge_aug == "random":
        head.append(RandomEdgeMask(prob=edge_aug_prob, max_frac=edge_max_frac, grey_fill=edge_grey_fill))
    head.append(fit)
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
                train=None, pin_memory=False, preprocess="resize", pad_aug="none",
                pad_grey_range=PAD_AUG_GREY_RANGE, edge_aug="none", edge_aug_prob=EDGE_AUG_PROB,
                edge_max_frac=EDGE_AUG_MAX_FRAC, edge_grey_fill=EDGE_AUG_GREY_FILL):
    """DataLoader for one split. `train` defaults to (split == 'train') and controls
    shuffling and augmentation. Returns (loader, frame)."""
    train = (split == "train") if train is None else train
    frame = load_split(manifest_path, data_root, split)
    ds = ChestXrayDataset(frame, build_transforms(train, preprocess, pad_aug, pad_grey_range,
                                                  edge_aug, edge_aug_prob, edge_max_frac, edge_grey_fill))
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
