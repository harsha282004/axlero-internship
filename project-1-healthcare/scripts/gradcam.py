"""Grad-CAM for the baseline CNN on VALIDATION images. Qualitative evidence only.

A heatmap shows where the model's prediction was sensitive; it is NOT proof that the
model reasons about lung pathology correctly, and it can look plausible for shortcuts.
Only the val split is read; the test split is never touched.

Run from project-1-healthcare/:
    .venv\\Scripts\\python.exe scripts\\gradcam.py --data-root C:\\Users\\harsh\\dataset\\kermany

Outputs go to runs/gradcam_baseline/ (git-ignored).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.dataset import MEAN, STD, make_loader
from src.evaluate import load_model
from src.train import predict
from src.utils import DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, load_json, resolve_data_root, set_seed

CATEGORIES = ["TP", "TN", "FP", "FN"]  # positive class = PNEUMONIA
TITLES = {"TP": "True PNEUMONIA, predicted PNEUMONIA", "TN": "True NORMAL, predicted NORMAL",
          "FP": "True NORMAL, predicted PNEUMONIA (false alarm)",
          "FN": "True PNEUMONIA, predicted NORMAL (missed)"}


def target_layer(model, name):
    """Last ReLU of the last conv block (before its max-pool): 28x28 maps for 224 inputs."""
    if name != "baseline":
        raise NotImplementedError(f"Grad-CAM is implemented/tested for the baseline CNN only, got {name!r}")
    return model.features[-1][5]


def grad_cam(model, layer, x):
    """x: [1,3,H,W]. Explains the PREDICTED side of the single logit (sign flipped for NORMAL)."""
    store = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(x).squeeze()
        prob = torch.sigmoid(logit).item()
        sign = 1.0 if prob >= model._threshold else -1.0
        (sign * logit).backward()
    finally:
        handle.remove()
    act, grad = store["act"].detach(), store["grad"].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return cam.cpu().numpy(), prob


def denorm(x):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x.cpu() * std + mean)[0].clamp(0, 1).numpy()  # grayscale replicated -> channel 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Grad-CAM on validation examples (qualitative; val only).")
    p.add_argument("--checkpoint", default=str(DEFAULT_OUTPUT_ROOT / "baseline_smoke" / "best.pt"))
    p.add_argument("--data-root", help="folder that CONTAINS chest_xray/ (or set CXR_DATA_ROOT)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "gradcam_baseline"))
    p.add_argument("--per-category", type=int, default=4, help="examples per TP/TN/FP/FN (seeded random sample)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args(argv)

    set_seed(args.seed)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    root = resolve_data_root(args.data_root)

    model, ckpt = load_model(ckpt_path, device)
    model.eval()
    layer = target_layer(model, ckpt["model"]["name"])
    thr_file = ckpt_path.parent / "threshold.json"
    model._threshold = load_json(thr_file)["threshold"]  # validation-selected threshold
    print(f"Loaded {ckpt['model']['name']} epoch {ckpt['epoch']}; val-selected threshold {model._threshold:.4f}")
    print("Validation split only; the test split is not read. Heatmaps are qualitative evidence only.")

    loader, frame = make_loader(args.manifest, root, "val", 64, num_workers=0, seed=args.seed,
                                train=False, pin_memory=device.type == "cuda")
    probs, targets, _ = predict(model, loader, device, amp=False, desc="val predictions")
    pred = (probs >= model._threshold).astype(int)
    cat = np.select([(targets == 1) & (pred == 1), (targets == 0) & (pred == 0),
                     (targets == 0) & (pred == 1), (targets == 1) & (pred == 0)], CATEGORIES, default="")
    print("Validation categories:", {c: int((cat == c).sum()) for c in CATEGORIES})

    rng = np.random.RandomState(args.seed)
    chosen = {c: sorted(rng.choice(np.flatnonzero(cat == c), min(args.per_category, (cat == c).sum()),
                                   replace=False).tolist()) for c in CATEGORIES}

    n = args.per_category
    fig, axes = plt.subplots(len(CATEGORIES), n, figsize=(2.6 * n, 2.8 * len(CATEGORIES)), squeeze=False)
    rows = []
    for r, c in enumerate(CATEGORIES):
        for k in range(n):
            ax = axes[r][k]
            ax.axis("off")
            if k >= len(chosen[c]):
                continue
            i = chosen[c][k]
            x, _ = loader.dataset[i]
            cam, prob = grad_cam(model, layer, x.unsqueeze(0).to(device))
            ax.imshow(denorm(x), cmap="gray")
            ax.imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
            ax.set_title(f"{c} p={prob:.2f}", fontsize=8)
            if k == 0:
                ax.text(-0.05, 0.5, c, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10)
            rows.append({"category": c, "val_index": i, "path": frame.rel_path[i], "patient_id": frame.patient_id[i],
                         "label": frame.label[i], "prob_pneumonia": float(prob), "threshold": model._threshold})
    fig.suptitle("Grad-CAM, baseline CNN, VALIDATION examples (seeded random sample per outcome).\n"
                 "Qualitative evidence only - not proof of correct reasoning. Educational project.", fontsize=9)
    fig.tight_layout(rect=(0.02, 0, 1, 0.93))
    fig.savefig(out_dir / "gradcam_grid.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "selected_examples.csv", index=False)
    print(f"Wrote {out_dir / 'gradcam_grid.png'} and selected_examples.csv ({len(rows)} images)")
    for c in CATEGORIES:
        print("  ", TITLES[c], "->", [Path(frame.rel_path[i]).name for i in chosen[c]])


if __name__ == "__main__":
    main()
