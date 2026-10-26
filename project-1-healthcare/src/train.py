"""Train a NORMAL-vs-PNEUMONIA classifier. Educational project, not a diagnostic tool.

Run from project-1-healthcare/:
    python -m src.train --data-root D:\\datasets\\kermany --model baseline --epochs 3

Model selection and the decision threshold use the VALIDATION split only. The test
split is never loaded here; use src.evaluate for the final, one-off test report.
"""
import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.dataset import (EDGE_AUG_FILL_CHOICES, EDGE_AUG_GREY_FILL, EDGE_AUG_MAX_FRAC, EDGE_AUG_MODES,
                         EDGE_AUG_OPS, EDGE_AUG_PROB, PAD_AUG_GREY_RANGE, PAD_AUG_KEEP_STANDARD_PROB,
                         PAD_AUG_MODES, PREPROCESS_MODES, class_counts, make_loader)
from src.metrics import THRESHOLD_STRATEGIES, compute_metrics, select_threshold
from src.models import MODEL_NAMES, build_model, count_parameters, set_train_mode
from src.utils import (DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, resolve_data_root,
                       save_json, set_seed, setup_logging)

log = logging.getLogger("cxr")

SELECT_METRICS = ("val_roc_auc", "val_pr_auc", "val_f1", "val_loss")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a chest X-ray NORMAL/PNEUMONIA classifier (educational use only). "
                    "Best epoch and threshold are chosen on the validation split; test is untouched.")
    p.add_argument("--data-root", help="folder that CONTAINS chest_xray/ (or set CXR_DATA_ROOT)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="splits.csv (default: data/splits.csv)")
    p.add_argument("--model", choices=MODEL_NAMES, default="baseline")
    p.add_argument("--pretrained", action="store_true",
                   help="densenet121 only: load ImageNet weights. DOWNLOADS them if not cached; off by default")
    p.add_argument("--freeze-backbone", action="store_true", help="densenet121 only: train the head only")
    p.add_argument("--preprocess", choices=PREPROCESS_MODES, default="resize",
                   help="resize: squash to 224x224 (default); pad: aspect-preserving resize + centred padding")
    p.add_argument("--pad-aug", choices=PAD_AUG_MODES, default="none",
                   help="TRAINING-only randomised padding (requires --preprocess pad). none = unchanged pipeline. "
                        "random: 50%% standard centred black; else fill in {black, image mean, random grey} "
                        "(equal prob.) and a random offset along the padded axis")
    p.add_argument("--pad-grey-range", type=int, nargs=2, metavar=("LO", "HI"), default=list(PAD_AUG_GREY_RANGE),
                   help="inclusive integer range for the random grey fill (default 64 192)")
    p.add_argument("--edge-aug", choices=EDGE_AUG_MODES, default="none",
                   help="TRAINING-only randomised edge crop/mask, independent of --pad-aug and of "
                        "--preprocess. none = unchanged pipeline (default). random: with probability "
                        "--edge-aug-prob, crop or mask (equal prob.) an outer border of width sampled "
                        "uniformly from 0 to --edge-max-frac of the shorter image side, filled with "
                        "black, a fixed grey, or this image's mean intensity (equal prob.)")
    p.add_argument("--edge-aug-prob", type=float, default=EDGE_AUG_PROB,
                   help="probability a training image receives an edge crop/mask at all (default 0.3)")
    p.add_argument("--edge-max-frac", type=float, default=EDGE_AUG_MAX_FRAC,
                   help="maximum edge crop/mask width as a fraction of min(width, height); must be in "
                        "[0, 0.10] (default 0.10)")
    p.add_argument("--edge-grey-fill", type=int, default=EDGE_AUG_GREY_FILL,
                   help="fixed grey fill value for edge masking, in [0, 255] (default 128)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="use ~1e-4 for pretrained densenet121")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no-class-weights", action="store_true", help="disable pos_weight for class imbalance")
    p.add_argument("--select-metric", choices=SELECT_METRICS, default="val_roc_auc",
                   help="validation metric used to pick the best epoch")
    p.add_argument("--patience", type=int, default=5, help="early-stop patience in epochs (0 = off)")
    p.add_argument("--threshold-strategy", choices=THRESHOLD_STRATEGIES, default="youden",
                   help="how to pick the decision threshold on validation data")
    p.add_argument("--target-sensitivity", type=float, default=0.95,
                   help="used by --threshold-strategy sensitivity")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--no-deterministic", action="store_true", help="allow faster, non-deterministic cuDNN")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT), help="parent of run folders (git-ignored)")
    p.add_argument("--run-name", help="run folder name (default: <model>_<timestamp>)")
    return p.parse_args(argv)


@torch.inference_mode()
def predict(model, loader, device, amp=False, criterion=None, desc="predict"):
    """Return (probs, targets, mean_loss or None) as numpy arrays."""
    model.eval()
    probs, targets, loss_sum, n = [], [], 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x).squeeze(1).float()
        if criterion is not None:
            loss_sum += criterion(logits, y.float().to(device)).item() * len(y)
        n += len(y)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets), (loss_sum / n if criterion is not None else None)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp, desc):
    set_train_mode(model)
    loss_sum, n = 0.0, 0
    for x, y in tqdm(loader, desc=desc, leave=False):
        x, y = x.to(device, non_blocking=True), y.float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x).squeeze(1)
        loss = criterion(logits.float(), y)
        if not math.isfinite(loss.item()):
            raise FloatingPointError("Training loss became NaN/inf; lower --lr or check the data")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        n += len(y)
    return loss_sum / n


def selection_score(name, val_metrics, val_loss):
    """Higher is better. Undefined metrics (single-class validation) score -inf."""
    if name == "val_loss":
        return -val_loss
    value = val_metrics[name.removeprefix("val_")]
    return -math.inf if value is None else value


def main(argv=None):
    args = parse_args(argv)
    if not (0.0 <= args.edge_aug_prob <= 1.0):
        raise ValueError(f"--edge-aug-prob must be in [0, 1], got {args.edge_aug_prob}")
    if not (0.0 <= args.edge_max_frac <= 0.10):
        raise ValueError(f"--edge-max-frac must be in [0, 0.10], got {args.edge_max_frac}")
    if not (0 <= args.edge_grey_fill <= 255):
        raise ValueError(f"--edge-grey-fill must be in [0, 255], got {args.edge_grey_fill}")
    run_name = args.run_name or f"{args.model}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(args.output_dir) / run_name
    if (run_dir / "best.pt").exists():
        raise FileExistsError(f"{run_dir} already contains a trained run; choose another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    log.info("Educational project: results are NOT clinically validated.")

    set_seed(args.seed, deterministic=not args.no_deterministic)
    device = get_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    data_root = resolve_data_root(args.data_root)
    log.info("run_dir=%s device=%s amp=%s preprocess=%s pad_aug=%s edge_aug=%s", run_dir, device, amp,
             args.preprocess, args.pad_aug, args.edge_aug)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    pin = device.type == "cuda"
    train_loader, train_df = make_loader(args.manifest, data_root, "train", args.batch_size,
                                         args.num_workers, args.seed, train=True, pin_memory=pin,
                                         preprocess=args.preprocess, pad_aug=args.pad_aug,
                                         pad_grey_range=tuple(args.pad_grey_range), edge_aug=args.edge_aug,
                                         edge_aug_prob=args.edge_aug_prob, edge_max_frac=args.edge_max_frac,
                                         edge_grey_fill=args.edge_grey_fill)
    val_loader, val_df = make_loader(args.manifest, data_root, "val", args.batch_size,
                                     args.num_workers, args.seed, train=False, pin_memory=pin,
                                     preprocess=args.preprocess)
    for name, df in (("train", train_df), ("val", val_df)):
        n0, n1 = class_counts(df)
        log.info("%s: %d images (NORMAL=%d, PNEUMONIA=%d)", name, len(df), n0, n1)
    n0, n1 = class_counts(train_df)
    if n0 == 0 or n1 == 0:
        raise ValueError("Training split must contain both classes")

    model = build_model(args.model, dropout=args.dropout, pretrained=args.pretrained,
                        freeze=args.freeze_backbone).to(device)
    total, trainable = count_parameters(model)
    log.info("model=%s params=%d trainable=%d", args.model, total, trainable)

    # Class imbalance: weight positives by n_neg / n_pos, computed on TRAIN labels only.
    pos_weight = None if args.no_class_weights else torch.tensor([n0 / n1], device=device)
    log.info("pos_weight=%s", None if pos_weight is None else round(pos_weight.item(), 4))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    config = {**vars(args), "run_dir": str(run_dir), "data_root": str(data_root), "device": str(device),
              "amp": amp, "pos_weight": None if pos_weight is None else pos_weight.item()}
    save_json(config, run_dir / "config.json")

    history, best_score, best_epoch, bad_epochs = [], -math.inf, 0, 0
    pad_aug_meta = {"mode": args.pad_aug, "keep_standard_prob": PAD_AUG_KEEP_STANDARD_PROB,
                    "fill_choices": ["black0", "image_mean", "random_grey"], "grey_range": list(args.pad_grey_range),
                    "random_offset": True, "applied_to": "train_only"}
    edge_aug_meta = {"mode": args.edge_aug, "prob": args.edge_aug_prob, "max_frac": args.edge_max_frac,
                     "ops": list(EDGE_AUG_OPS), "fill_choices": list(EDGE_AUG_FILL_CHOICES),
                     "grey_fill": args.edge_grey_fill, "applied_to": "train_only",
                     "independent_of": "pad_aug"}
    ckpt_meta = {"name": args.model, "dropout": args.dropout, "preprocess": args.preprocess,
                 "pad_aug": pad_aug_meta, "edge_aug": edge_aug_meta}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp,
                                     f"epoch {epoch}/{args.epochs} train")
        probs, targets, val_loss = predict(model, val_loader, device, amp, criterion,
                                           f"epoch {epoch}/{args.epochs} val")
        m = compute_metrics(targets, probs, 0.5)
        scheduler.step(val_loss)
        score = selection_score(args.select_metric, m, val_loss)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_roc_auc": m["roc_auc"], "val_pr_auc": m["pr_auc"], "val_f1@0.5": m["f1"],
               "val_sensitivity@0.5": m["sensitivity"], "val_specificity@0.5": m["specificity"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1)}
        history.append(row)
        log.info("epoch %d/%d train_loss=%.4f val_loss=%.4f val_auc=%s sens@0.5=%s spec@0.5=%s (%.0fs)",
                 epoch, args.epochs, train_loss, val_loss, _fmt(m["roc_auc"]),
                 _fmt(m["sensitivity"]), _fmt(m["specificity"]), row["seconds"])
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model": ckpt_meta, "state_dict": model.state_dict(), "epoch": epoch,
                        "select_metric": args.select_metric, "select_value": score,
                        "seed": args.seed}, run_dir / "best.pt")
            log.info("  new best on %s = %.4f -> saved best.pt", args.select_metric, score)
        else:
            bad_epochs += 1
            if args.patience and bad_epochs >= args.patience:
                log.info("Early stopping: no %s improvement for %d epochs", args.select_metric, bad_epochs)
                break
    save_json(history, run_dir / "history.json")
    if not (run_dir / "best.pt").exists():
        raise RuntimeError(f"No checkpoint saved: {args.select_metric} was undefined in every epoch "
                           "(does the validation split contain both classes?)")

    # Threshold: chosen on validation predictions of the best checkpoint.
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    probs, targets, _ = predict(model, val_loader, device, amp, desc="val (best epoch)")
    threshold, info = select_threshold(targets, probs, args.threshold_strategy, args.target_sensitivity)
    val_metrics = compute_metrics(targets, probs, threshold)
    save_json({"threshold": threshold, "selected_on": "val", "best_epoch": ckpt["epoch"],
               "select_metric": args.select_metric, **info}, run_dir / "threshold.json")
    save_json(val_metrics, run_dir / "val_metrics.json")
    log.info("Best epoch %d (%s=%.4f). Threshold %.4f selected on VALIDATION (%s): "
             "val sensitivity=%s specificity=%s",
             ckpt["epoch"], args.select_metric, ckpt["select_value"], threshold, args.threshold_strategy,
             _fmt(val_metrics["sensitivity"]), _fmt(val_metrics["specificity"]))
    log.info("Test split was not used. Final report: python -m src.evaluate --checkpoint %s "
             "--data-root <root> --split test", run_dir / "best.pt")
    return run_dir


def _fmt(v):
    return "n/a" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()
