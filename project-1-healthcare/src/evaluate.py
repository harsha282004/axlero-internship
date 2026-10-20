"""Evaluate a trained checkpoint on one split. Educational project, not a diagnostic tool.

Run from project-1-healthcare/:
    python -m src.evaluate --checkpoint runs/<run>/best.pt --data-root D:\\datasets\\kermany --split test

The threshold comes from <run>/threshold.json (chosen on validation data by src.train).
Use the test split for the final report only; do not tune anything based on it.
"""
import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.dataset import class_counts, make_loader
from src.metrics import compute_metrics, select_threshold, THRESHOLD_STRATEGIES
from src.models import build_model
from src.train import predict
from src.utils import (CLASS_NAMES, DEFAULT_MANIFEST, SPLITS, get_device, load_json,
                       resolve_data_root, save_json, set_seed, setup_logging)

log = logging.getLogger("cxr")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a split (default: test). Threshold defaults to the "
                    "validation-selected value saved by src.train.")
    p.add_argument("--checkpoint", required=True, help="path to best.pt from src.train")
    p.add_argument("--data-root", help="folder that CONTAINS chest_xray/ (or set CXR_DATA_ROOT)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--split", choices=SPLITS, default="test")
    p.add_argument("--threshold", type=float,
                   help="override the saved threshold (must NOT be tuned on the test split)")
    p.add_argument("--threshold-from-val", choices=THRESHOLD_STRATEGIES,
                   help="ignore threshold.json and re-select the threshold on the val split")
    p.add_argument("--target-sensitivity", type=float, default=0.95)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--output-dir", help="where to write results (default: <run>/eval_<split>)")
    return p.parse_args(argv)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    meta = ckpt["model"]
    model = build_model(meta["name"], dropout=meta["dropout"], pretrained=False)  # weights come from ckpt
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ckpt


def plot_confusion_matrix(cm, path, title):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    run_dir = ckpt_path.parent
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "evaluate.log")
    log.info("Educational project: results are NOT clinically validated.")
    set_seed(args.seed)
    device = get_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    data_root = resolve_data_root(args.data_root)
    pin = device.type == "cuda"

    model, ckpt = load_model(ckpt_path, device)
    preprocess = ckpt["model"].get("preprocess", "resize")  # old checkpoints predate this option
    log.info("Loaded %s from %s (epoch %s, preprocess=%s)", ckpt["model"]["name"], ckpt_path,
             ckpt.get("epoch"), preprocess)

    # Threshold source: explicit override > re-selection on val > saved threshold.json (from val).
    if args.threshold is not None:
        threshold, source = args.threshold, "command-line override"
        if args.split == "test":
            log.warning("Using a manual threshold on the test split; make sure it was not tuned on test data.")
    elif args.threshold_from_val:
        vloader, _ = make_loader(args.manifest, data_root, "val", args.batch_size, args.num_workers,
                                 args.seed, train=False, pin_memory=pin, preprocess=preprocess)
        vp, vt, _ = predict(model, vloader, device, amp, desc="val (threshold)")
        threshold, _ = select_threshold(vt, vp, args.threshold_from_val, args.target_sensitivity)
        source = f"re-selected on val ({args.threshold_from_val})"
    else:
        tfile = run_dir / "threshold.json"
        if not tfile.is_file():
            raise FileNotFoundError(f"{tfile} not found; pass --threshold or --threshold-from-val")
        info = load_json(tfile)
        threshold, source = info["threshold"], f"{tfile.name} (selected on {info.get('selected_on')})"
    log.info("Decision threshold = %.4f [%s]", threshold, source)

    loader, frame = make_loader(args.manifest, data_root, args.split, args.batch_size, args.num_workers,
                                args.seed, train=False, pin_memory=pin, preprocess=preprocess)
    n0, n1 = class_counts(frame)
    log.info("Evaluating split=%s: %d images (NORMAL=%d, PNEUMONIA=%d)", args.split, len(frame), n0, n1)
    probs, targets, _ = predict(model, loader, device, amp, desc=f"{args.split}")
    metrics = compute_metrics(targets, probs, threshold)
    metrics.update({"split": args.split, "checkpoint": str(ckpt_path), "threshold_source": source,
                    "disclaimer": "Educational project; not clinically validated."})

    save_json(metrics, out_dir / "metrics.json")
    preds = pd.DataFrame({"path": frame.rel_path, "patient_id": frame.patient_id, "label": targets,
                          "prob_pneumonia": probs, "pred": (probs >= threshold).astype(int)})
    preds.to_csv(out_dir / "predictions.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], out_dir / "confusion_matrix.png",
                          f"{args.split} @ thr={threshold:.3f}")

    log.info("Confusion matrix (rows=true NORMAL/PNEUMONIA, cols=pred): %s", metrics["confusion_matrix"])
    for k in ("sensitivity", "specificity", "precision", "f1", "roc_auc", "pr_auc", "accuracy"):
        v = metrics[k]
        log.info("  %-12s %s", k, "n/a" if v is None else f"{v:.4f}")
    log.info("\n%s", metrics["classification_report"])
    log.info("Results written to %s", out_dir)
    return metrics


if __name__ == "__main__":
    main()
