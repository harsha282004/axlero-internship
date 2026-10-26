"""Grad-CAM audit of the three DenseNet121 checkpoints (VALIDATION images only, no training).

Qualitative evidence only: a heatmap shows where the prediction was sensitive; it is NOT proof of
anatomical localisation or correct reasoning and can look plausible for shortcuts. The test split is
never read.

Target layer: the output of model.features (DenseNet121 `norm5`, 1024 channels, 7x7 for 224 inputs) after
ReLU, which is exactly the tensor torchvision's forward global-average-pools into the classifier. A runtime
check confirms classifier(GAP(ReLU(features))) reproduces the model logit. The CAM explains the PREDICTED
side of the single logit (sign flipped for NORMAL predictions). 7x7 maps are coarse: one cell = 32x32 px.

Run from project-1-healthcare/:
    .venv\\Scripts\\python.exe scripts\\gradcam_densenet.py --data-root C:\\Users\\harsh\\dataset\\kermany
Outputs: runs/gradcam_densenet/ (git-ignored).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import size_matched_stress_test as base          # existing, unchanged (auc helper)
import size_matched_stress_test_all as allm      # existing, unchanged (alignment check)
from gradcam import denorm                       # existing, unchanged
from src.dataset import IMAGE_SIZE, make_loader, load_split
from src.evaluate import load_model
from src.train import predict
from src.utils import (DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, load_json, resolve_data_root,
                       save_json, set_seed)

RUNS = {42: "densenet121_pad_seed42", 43: "densenet121_pad_seed43", 44: "densenet121_pad_seed44"}
AUC_TOL = 1e-4
STRATA = ["TP-high", "TP-low", "TN-high", "TN-low", "FP", "FN"]
ALPHA = 0.4


class GradCAM:
    """Grad-CAM on model.features output (after ReLU), batched; BN is in eval mode so samples are independent."""

    def __init__(self, model):
        self.model, self.act, self.grad = model, None, None
        self.handle = model.features.register_forward_hook(self._hook)

    def _hook(self, _m, _inp, out):
        a = F.relu(out)  # same values the classifier path uses; not in-place so the grad hook stays valid
        a.register_hook(lambda g: setattr(self, "grad", g))
        self.act = a
        return a.clone()  # torchvision's forward then applies an in-place ReLU on the clone

    def close(self):
        self.handle.remove()

    def __call__(self, x, signs, check_logits=False):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)
        if check_logits:  # verify the hooked tensor really feeds the classifier
            manual = self.model.classifier(F.adaptive_avg_pool2d(F.relu(self.act.detach()), 1).flatten(1)).squeeze(1)
            if not torch.allclose(manual, logits.detach(), atol=1e-4):
                raise RuntimeError("hooked layer does not reproduce the model logit; wrong Grad-CAM target")
        (signs * logits).sum().backward()
        act, grad = self.act.detach(), self.grad.detach()
        cam = F.relu((grad.mean(dim=(2, 3), keepdim=True) * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mx = cam.flatten(1).max(dim=1).values.clamp_min(1e-12).view(-1, 1, 1)
        return (cam / mx).cpu().numpy(), logits.detach().cpu().numpy()


def content_geometry(w, h, size=IMAGE_SIZE):
    """Where the image sits inside the padded canvas (mirrors LetterboxSquare)."""
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    return ox, oy, nw, nh


def region_masks(w, h, size=IMAGE_SIZE):
    ox, oy, nw, nh = content_geometry(w, h, size)
    content = np.zeros((size, size), bool)
    content[oy:oy + nh, ox:ox + nw] = True
    yy, xx = np.mgrid[:size, :size]
    d_edge = np.minimum.reduce([xx - ox, ox + nw - 1 - xx, yy - oy, oy + nh - 1 - yy])
    band = max(1, round(0.10 * min(nw, nh)))
    edge = content & (d_edge < band)
    cs = max(1, round(0.20 * min(nw, nh)))
    corner = content & (((xx - ox < cs) | (ox + nw - 1 - xx < cs)) & ((yy - oy < cs) | (oy + nh - 1 - yy < cs)))
    return content, ~content, edge, corner


def attention_metrics(cam, w, h):
    """Mass of the heatmap in padding / image-edge band / image corners, relative to the area fraction.

    enrichment = mass_fraction / area_fraction (1.0 = uniform; >1 = attention concentrated there)."""
    content, pad, edge, corner = region_masks(w, h)
    total = cam.sum()
    if total <= 0:
        return {k: float("nan") for k in ("pad_mass", "pad_enrich", "edge_enrich", "corner_enrich")} | {
            "peak_in_pad": False, "peak_in_edge": False, "peak_in_corner": False, "cam_empty": True}
    c_total = (cam * content).sum()
    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    out = {"pad_mass": float((cam * pad).sum() / total),
           "pad_enrich": float((cam * pad).sum() / total / pad.mean()) if pad.any() else float("nan"),
           "edge_enrich": float((cam * edge).sum() / c_total / (edge.sum() / content.sum())) if c_total > 0 else float("nan"),
           "corner_enrich": float((cam * corner).sum() / c_total / (corner.sum() / content.sum())) if c_total > 0 else float("nan"),
           "peak_in_pad": bool(pad[py, px]), "peak_in_edge": bool(edge[py, px]), "peak_in_corner": bool(corner[py, px]),
           "cam_empty": False}
    return out


def overlay(gray, cam):
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - ALPHA) * np.stack([gray] * 3, -1) + ALPHA * heat, 0, 1)


def stratify(df, thr, n_correct, n_err, rng):
    """Reproducible stratified sample for one checkpoint: confidence terciles inside TP and TN, plus errors."""
    chosen = {}
    p_pred = np.where(df.pred == 1, df.prob, 1 - df.prob)
    for cat in ("TP", "TN"):
        idx = np.flatnonzero((df.category == cat).values)
        if len(idx) == 0:
            continue
        order = idx[np.argsort(p_pred[idx])]
        third = max(1, len(order) // 3)
        for name, pool in ((f"{cat}-low", order[:third]), (f"{cat}-high", order[-third:])):
            chosen[name] = sorted(rng.choice(pool, min(n_correct, len(pool)), replace=False).tolist())
    for cat in ("FP", "FN"):
        idx = np.flatnonzero((df.category == cat).values)
        chosen[cat] = sorted(rng.choice(idx, min(n_err, len(idx)), replace=False).tolist()) if len(idx) else []
    return chosen


def main(argv=None):
    p = argparse.ArgumentParser(description="Validation-only Grad-CAM audit for DenseNet121 seeds 42-44.")
    p.add_argument("--data-root")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--audit-dir", default=str(DEFAULT_OUTPUT_ROOT / "shortcut_audit"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "gradcam_densenet"))
    p.add_argument("--per-correct-stratum", type=int, default=3)
    p.add_argument("--per-error-stratum", type=int, default=4)
    p.add_argument("--shared-per-class", type=int, default=4)
    p.add_argument("--max-hard", type=int, default=8)
    p.add_argument("--cam-batch", type=int, default=8)
    p.add_argument("--seed", type=int, default=42, help="sampling seed")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args(argv)

    set_seed(args.seed)
    root = resolve_data_root(args.data_root)
    out = Path(args.output_dir)
    for sub in ("originals", "heatmaps", "overlays", "panels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print("Validation split only; the test split is not read. Qualitative evidence only.")

    # ---- alignment + geometry metadata
    meta = pd.read_csv(Path(args.audit_dir) / "metadata_val.csv")
    train_meta = pd.read_csv(Path(args.audit_dir) / "metadata_train.csv")
    frame = load_split(args.manifest, root, "val")
    allm.check_alignment(meta, train_meta, frame)
    order = frame.rel_path.tolist()
    meta = meta.set_index("path").loc[order].reset_index()
    print(f"Alignment OK: {len(frame)} val images match the manifest by path, label and patient_id.")

    # ---- phase 1: load each checkpoint, verify config, predict (same path as saved metrics), stratify
    models, preds, align = {}, {}, []
    loaders = {}
    for seed, run in RUNS.items():
        run_dir = DEFAULT_OUTPUT_ROOT / run
        cfg = load_json(run_dir / "config.json")
        model, ckpt = load_model(run_dir / "best.pt", device)
        pre = ckpt["model"].get("preprocess", "resize")
        if (ckpt["model"]["name"] != "densenet121" or cfg["model"] != "densenet121" or pre != "pad"
                or cfg["preprocess"] != pre or ckpt.get("seed") != cfg["seed"] or cfg["seed"] != seed
                or cfg.get("pretrained")):
            raise RuntimeError(f"{run}: unexpected configuration")
        model.eval()
        loader, fr = make_loader(args.manifest, root, "val", 32, num_workers=0, seed=args.seed, train=False,
                                 pin_memory=device.type == "cuda", preprocess=pre)
        if fr.rel_path.tolist() != order:
            raise RuntimeError(f"{run}: loader order differs from metadata order")
        prob, targets, _ = predict(model, loader, device, amp=device.type == "cuda", desc=f"s{seed} predict")
        if not (targets == meta.target.values).all():
            raise RuntimeError(f"{run}: label order mismatch")
        pooled, saved = base.auc_or_nan(targets, prob), load_json(run_dir / "val_metrics.json")["roc_auc"]
        if abs(pooled - saved) > AUC_TOL:
            raise RuntimeError(f"{run}: pooled val AUC {pooled:.6f} != saved {saved:.6f}")
        thr = load_json(run_dir / "threshold.json")["threshold"]
        pred = (prob >= thr).astype(int)
        df = pd.DataFrame({"val_index": np.arange(len(order)), "path": order, "patient_id": meta.patient_id,
                           "label": meta.label, "target": targets, "prob": prob, "pred": pred})
        df["category"] = np.select([(df.target == 1) & (df.pred == 1), (df.target == 0) & (df.pred == 0),
                                    (df.target == 0) & (df.pred == 1), (df.target == 1) & (df.pred == 0)],
                                   ["TP", "TN", "FP", "FN"], default="")
        models[seed], preds[seed], loaders[seed] = model, (df, thr), loader
        align.append({"seed": seed, "run": run, "preprocess": pre, "pooled_auc_here": pooled,
                      "pooled_auc_saved": saved, "threshold": thr, **df.category.value_counts().to_dict()})
        print(f"seed {seed} ({run}): densenet121/{pre}; pooled val AUC {pooled:.6f} vs saved {saved:.6f} OK; "
              f"thr {thr:.4f}; categories {df.category.value_counts().to_dict()}")

    # ---- sampling plan (seeded): per-seed strata, shared images, hard cases (misclassified by >=1 seed)
    rng = np.random.RandomState(args.seed)
    plan = {s: stratify(preds[s][0], preds[s][1], args.per_correct_stratum, args.per_error_stratum, rng) for s in RUNS}
    shared = []
    for lab in (0, 1):
        pool = np.flatnonzero(meta.target.values == lab)
        shared += sorted(rng.choice(pool, args.shared_per_class, replace=False).tolist())
    wrong_any = sorted(set().union(*[set(np.flatnonzero(preds[s][0].category.isin(["FP", "FN"]).values)) for s in RUNS]))
    hard = sorted(rng.choice(wrong_any, min(args.max_hard, len(wrong_any)), replace=False).tolist()) if wrong_any else []
    print(f"\nSampling plan (seeded {args.seed}): shared={len(shared)} images, hard cases={len(hard)} of {len(wrong_any)} "
          f"misclassified by at least one seed")
    for s in RUNS:
        print(f"  seed {s}:", {k: len(v) for k, v in plan[s].items()})

    # ---- phase 2: Grad-CAM for ALL validation images per seed (metrics), keep CAMs for the images we draw
    needed = {s: sorted(set(sum(plan[s].values(), [])) | set(shared) | set(hard)) for s in RUNS}
    cams, metrics_rows, gray_cache = {}, [], {}
    for seed in RUNS:
        model, (df, thr) = models[seed], preds[seed]
        sign_all = np.where(df.pred.values == 1, 1.0, -1.0)
        gc = GradCAM(model)
        ds, n = loaders[seed].dataset, len(df)
        first = True
        for start in tqdm(range(0, n, args.cam_batch), desc=f"s{seed} grad-cam", leave=False):
            idx = list(range(start, min(start + args.cam_batch, n)))
            x = torch.stack([ds[i][0] for i in idx]).to(device)
            cam, logits = gc(x, torch.tensor(sign_all[idx], dtype=torch.float32, device=device), check_logits=first)
            first = False
            for j, i in enumerate(idx):
                m = attention_metrics(cam[j], int(meta.width[i]), int(meta.height[i]))
                metrics_rows.append({"seed": seed, "val_index": i, "logit_fp32": float(logits[j]), **m})
                if i in needed[seed]:
                    cams[(seed, i)] = cam[j].astype(np.float32)
                if i in needed[seed] or i in shared or i in hard:
                    gray_cache[i] = denorm(x[j].detach())
        gc.close()
        print(f"seed {seed}: Grad-CAM computed for all {n} validation images; layer check passed")
        torch.cuda.empty_cache() if device.type == "cuda" else None

    met = pd.DataFrame(metrics_rows)
    allpred = pd.concat([preds[s][0].assign(seed=s, threshold=preds[s][1]) for s in RUNS])
    full = allpred.merge(met, on=["seed", "val_index"])
    full["fp32_pred_agrees"] = ((full.logit_fp32 >= np.log(full.threshold / (1 - full.threshold))) == (full.pred == 1))
    full.to_csv(out / "attention_metrics_all_val.csv", index=False)
    print("\nfp32-vs-AMP prediction agreement on the CAM pass:", full.groupby("seed").fp32_pred_agrees.mean().round(4).to_dict())

    summ = (full.groupby(["seed", "category"]).agg(
        n=("val_index", "count"), pad_mass=("pad_mass", "mean"), pad_enrich=("pad_enrich", "mean"),
        edge_enrich=("edge_enrich", "mean"), corner_enrich=("corner_enrich", "mean"),
        peak_in_pad=("peak_in_pad", "mean"), peak_in_edge=("peak_in_edge", "mean"),
        peak_in_corner=("peak_in_corner", "mean"), cam_empty=("cam_empty", "mean")).reset_index())
    summ.round(4).to_csv(out / "attention_summary_by_seed_category.csv", index=False)
    overall = (full.groupby("seed").agg(pad_mass=("pad_mass", "mean"), pad_enrich=("pad_enrich", "mean"),
               edge_enrich=("edge_enrich", "mean"), corner_enrich=("corner_enrich", "mean"),
               peak_in_pad=("peak_in_pad", "mean"), peak_in_edge=("peak_in_edge", "mean"),
               peak_in_corner=("peak_in_corner", "mean"), cam_empty=("cam_empty", "mean")).reset_index())
    # reference: share of each region by area (mean over validation images)
    areas = np.array([[m.mean() for m in region_masks(int(w), int(h))[1:]]  # pad, edge band, corners
                      for w, h in zip(meta.width, meta.height)])
    print("\nMean area fractions over validation images (padding / image-edge band / image corners, of the canvas):",
          dict(zip(["pad", "edge", "corner"], areas.mean(0).round(3))))
    print("\nAttention metrics over ALL validation images, per seed (enrichment 1.0 = uniform attention):")
    print(overall.round(3).to_string(index=False))
    print("\nBy seed and outcome category:")
    print(summ.round(3).to_string(index=False))
    overall.round(4).to_csv(out / "attention_summary_by_seed.csv", index=False)
    # label-wise view (are NORMAL and PNEUMONIA treated differently?)
    by_label = full.groupby(["seed", "label"]).agg(pad_mass=("pad_mass", "mean"), edge_enrich=("edge_enrich", "mean"),
                                                    corner_enrich=("corner_enrich", "mean"),
                                                    peak_in_edge=("peak_in_edge", "mean"),
                                                    peak_in_corner=("peak_in_corner", "mean")).reset_index()
    by_label.round(4).to_csv(out / "attention_summary_by_seed_label.csv", index=False)
    print("\nBy seed and TRUE label:")
    print(by_label.round(3).to_string(index=False))

    # ---- write per-image files + index for every drawn image
    index_rows = []
    strata_of = {}
    for s in RUNS:
        for name, ids in plan[s].items():
            for i in ids:
                strata_of.setdefault((s, i), []).append(name)
        for i in shared:
            strata_of.setdefault((s, i), []).append("shared")
        for i in hard:
            strata_of.setdefault((s, i), []).append("hard")
    for (s, i), names in sorted(strata_of.items()):
        df, thr = preds[s]
        r = df.iloc[i]
        cam, gray = cams[(s, i)], gray_cache[i]
        stem = f"val{i:03d}"
        orig_p, heat_p, over_p = out / "originals" / f"{stem}.png", out / "heatmaps" / f"seed{s}_{stem}.png", out / "overlays" / f"seed{s}_{stem}.png"
        plt.imsave(orig_p, gray, cmap="gray", vmin=0, vmax=1)
        plt.imsave(heat_p, cm.jet(cam))
        plt.imsave(over_p, overlay(gray, cam))
        mrow = full[(full.seed == s) & (full.val_index == i)].iloc[0]
        index_rows.append({"seed": s, "val_index": i, "path": r.path, "patient_id": r.patient_id, "true_label": r.label,
                           "pred_label": "PNEUMONIA" if r.pred else "NORMAL", "prob_pneumonia": float(r.prob),
                           "threshold": thr, "category": r.category, "strata": ";".join(names),
                           "pad_enrich": mrow.pad_enrich, "edge_enrich": mrow.edge_enrich,
                           "corner_enrich": mrow.corner_enrich, "peak_in_edge": bool(mrow.peak_in_edge),
                           "peak_in_corner": bool(mrow.peak_in_corner), "peak_in_pad": bool(mrow.peak_in_pad),
                           "original_png": str(orig_p.relative_to(out)), "heatmap_png": str(heat_p.relative_to(out)),
                           "overlay_png": str(over_p.relative_to(out))})
    pd.DataFrame(index_rows).to_csv(out / "gradcam_index.csv", index=False)

    # ---- panels
    def cell(ax, s, i, title=None):
        df, thr = preds[s]
        r = df.iloc[i]
        ax.imshow(overlay(gray_cache[i], cams[(s, i)]))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title or f"{r.category} p={r.prob:.2f}", fontsize=7)

    for s in RUNS:  # per-seed stratified grid
        ncol = max(args.per_correct_stratum, args.per_error_stratum)
        fig, axes = plt.subplots(len(STRATA), ncol, figsize=(2.3 * ncol, 2.5 * len(STRATA)), squeeze=False)
        for r_, st in enumerate(STRATA):
            for c_ in range(ncol):
                ax = axes[r_][c_]
                ax.axis("off")
                ids = plan[s].get(st, [])
                if c_ < len(ids):
                    ax.axis("on")
                    cell(ax, s, ids[c_], f"{preds[s][0].iloc[ids[c_]].label[:4]} p={preds[s][0].iloc[ids[c_]].prob:.2f}")
                elif c_ == 0:
                    ax.text(0.5, 0.5, f"{st}: none available", ha="center", va="center", fontsize=8)
            axes[r_][0].set_ylabel(st, fontsize=9)
        fig.suptitle(f"DenseNet121 pad seed {s}, VALIDATION, stratified seeded sample (rows: outcome/confidence). "
                     "Qualitative only.", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / "panels" / f"gradcam_seed{s}_stratified.png", dpi=130)
        plt.close(fig)

    def multi_seed_panel(ids, title, path):
        fig, axes = plt.subplots(len(ids), 4, figsize=(9.6, 2.4 * len(ids)), squeeze=False)
        for r_, i in enumerate(ids):
            axes[r_][0].imshow(gray_cache[i], cmap="gray", vmin=0, vmax=1)
            axes[r_][0].set_title(f"val{i:03d} true {meta.label[i][:4]}", fontsize=7)
            for c_, s in enumerate(RUNS, start=1):
                r = preds[s][0].iloc[i]
                cell(axes[r_][c_], s, i, f"s{s} {r.category} p={r.prob:.2f}")
            for ax in axes[r_]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(path, dpi=130)
        plt.close(fig)

    multi_seed_panel(shared, "Same seeded random validation images across the three seeds (4 NORMAL, 4 PNEUMONIA). "
                     "Qualitative only.", out / "panels" / "gradcam_shared_images_3seeds.png")
    if hard:
        multi_seed_panel(hard, "Validation images misclassified by at least one seed (seeded sample). Qualitative only.",
                         out / "panels" / "gradcam_hard_cases_3seeds.png")
    save_json({"note": "Qualitative Grad-CAM audit; validation only; test split never read.",
               "target_layer": "model.features output (norm5) after ReLU: 1024 x 7 x 7", "alignment": align,
               "sample_plan": {str(s): {k: v for k, v in plan[s].items()} for s in RUNS},
               "shared_indices": shared, "hard_indices": hard}, out / "summary.json")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
