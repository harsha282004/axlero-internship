"""Shared constants and helpers: seeding, device, logging, paths, JSON."""
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "splits.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs"  # git-ignored (see .gitignore)
DATA_ROOT_ENV = "CXR_DATA_ROOT"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
LABEL_TO_INDEX = {"NORMAL": 0, "PNEUMONIA": 1}
SPLITS = ("train", "val", "test")

log = logging.getLogger("cxr")


def set_seed(seed, deterministic=True):
    """Seed python, numpy and torch. Some CUDA ops can stay non-deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id):
    """DataLoader worker_init_fn: derive python/numpy seeds from the torch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_device(choice="auto"):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available in this PyTorch build")
    return torch.device(choice)


def setup_logging(log_file=None, level=logging.INFO):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def close_logging():
    """Close and detach log handlers (releases the log file; needed on Windows before cleanup)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        h.close()
        root.removeHandler(h)


def resolve_data_root(arg):
    """Dataset root = folder that CONTAINS the outer chest_xray/ folder."""
    value = arg or os.environ.get(DATA_ROOT_ENV)
    if not value:
        raise ValueError(f"Dataset root not given: pass --data-root or set {DATA_ROOT_ENV}. "
                         "It must be the folder that contains chest_xray/.")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is not a directory: {root}")
    return root


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
