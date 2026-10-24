"""Perturbation audit of the three DenseNet121 checkpoints (VALIDATION only, inference only).

Question: how much do predictions depend on the artificial padding and the outer image margins? A change
is NOT proof of shortcut use: masking can create out-of-distribution inputs and can remove real anatomy.
The test split is never read; nothing is trained; checkpoints are never modified.

Coordinate conventions: original image pixel coordinates (x to the right, y down), size w x h, BEFORE any
resizing. All perturbations are applied to the 8-bit grayscale version of the original image (the model's
saved preprocessing begins with a grayscale conversion anyway), then the checkpoint's exact saved pipeline
follows: Grayscale(3) -> LetterboxSquare(224, fill) -> ToTensor -> ImageNet Normalize.
  v           = round(mean pixel value of the whole original grayscale image), same value in all channels
  margin(f)   = pixels with x < m or x >= w-m or y < m or y >= h-m, m = round(f * min(w, h)), set to v
  ring        = band of width m = round(0.10 * min(w,h)) located between insets a = round(0.20 * min(w,h))
                and a + m, set to v (INTERIOR reference: removes real anatomy, similar masked area)
  padding     = the artificial letterbox bars added by the pipeline; their geometry follows exactly from
                (w, h): scale = 224 / max(w, h), content box nw x nh centred at ((224-nw)//2, (224-nh)//2).
                The pad value is normally 0 (black); conditions below change the fill value.
Conditions:
  control              : unmodified pipeline (must reproduce the saved predictions)
  pad_imgmean          : padding filled with v (neutral, image-brightness-matched)
  pad_gray128          : padding filled with 128 (constant mid-grey; fill-sensitivity check)
  margin5_imgmean      : 5% outer margin set to v
  margin10_imgmean     : 10% outer margin set to v
  ring10_interior      : interior ring (see above) set to v (reference for masking real anatomy)
  pad_margin10_imgmean : 10% margin set to v AND padding filled with v
Marker/text regions are NOT masked: no annotations exist and automatic detection would be a guess.
Decision thresholds are each checkpoint's ORIGINAL saved validation threshold (never retuned).

Run from project-1-healthcare/:
    .venv\\Scripts\\python.exe scripts\\perturbation_audit_densenet.py --data-root C:\\Users\\harsh\\dataset\\kermany
Outputs: runs/perturbation_audit_densenet/ (git-ignored).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm import tqdm

import size_matched_stress_test_all as allm          # existing, unchanged (alignment check)
from gradcam import denorm                           # existing, unchanged
from gradcam_densenet import region_masks            # existing, unchanged (pad geometry)
from src.dataset import IMAGE_SIZE, MEAN, STD, ChestXrayDataset, LetterboxSquare, load_split, make_loader
from src.evaluate import load_model
from src.train import predict
from src.utils import (DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, load_json, resolve_data_root,
                       save_json, set_seed)

RUNS = {42: "densenet121_pad_seed42", 43: "densenet121_pad_seed43", 44: "densenet121_pad_seed44"}
CONDITIONS = ["control", "pad_imgmean", "pad_gray128", "margin5_imgmean", "margin10_imgmean",
              "ring10_interior", "pad_margin10_imgmean"]
AUC_TOL = 1e-4
TENSOR_TOL = 1e-6
GRAY3 = transforms.Grayscale(num_output_channels=3)


def load_gray(path):
    with Image.open(path) as im:
        im.load()
        return im.convert("L")


def mask_margin(gray, frac, v):
    w, h = gray.size
    m = round(frac * min(w, h))
    a = np.array(gray)
    if m > 0:
        a[:m], a[h - m:], a[:, :m], a[:, w - m:] = v, v, v, v
    return Image.fromarray(a)


def mask_ring(gray, inset_frac, width_frac, v):
    w, h = gray.size
    s = min(w, h)
    a0, m = round(inset_frac * s), round(width_frac * s)
    outer, inner = np.zeros((h, w), bool), np.zeros((h, w), bool)
    outer[a0:h - a0, a0:w - a0] = True
    inner[a0 + m:h - a0 - m, a0 + m:w - a0 - m] = True
    a = np.array(gray)
    a[outer & ~inner] = v
    return Image.fromarray(a)


def apply_condition(gray, cond):
    """Return (perturbed grayscale image, padding fill value)."""
    v = int(round(np.asarray(gray).mean()))
    if cond == "control":
        return gray, 0
    if cond == "pad_imgmean":
        return gray, v
    if cond == "pad_gray128":
        return gray, 128
    if cond == "margin5_imgmean":
        return mask_margin(gray, 0.05, v), 0
    if cond == "margin10_imgmean":
        return mask_margin(gray, 0.10, v), 0
    if cond == "ring10_interior":
        return mask_ring(gray, 0.20, 0.10, v), 0
    if cond == "pad_margin10_imgmean":
        return mask_margin(gray, 0.10, v), v
    raise ValueError(cond)


def to_tensor(gray, cond):
    g, fill = apply_condition(gray, cond)
    img = LetterboxSquare(IMAGE_SIZE, fill=fill)(GRAY3(g))
    return TF.normalize(TF.to_tensor(img), MEAN, STD)


def metric_set(y, p, thr):
    pred = p >= thr
    tp, tn = int((pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    fp, fn = int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    two = len(np.unique(y)) == 2
    div = lambda a, b: a / b if b else float("nan")
    return {"auc": float(roc_auc_score(y, p)) if two else float("nan"),
            "pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp), "precision": div(tp, tp + fp),
            "f1": div(2 * tp, 2 * tp + fp + fn), "accuracy": (tp + tn) / len(y), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def paired_bootstrap(y, patient, p_cond, p_ctrl, thr, n_boot, seed):
    """Patient-clustered paired bootstrap of metric(condition) - metric(control)."""
    keys = ["auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "accuracy"]
    ug, inv = np.unique(patient, return_inverse=True)
    by_group = [np.flatnonzero(inv == i) for i in range(len(ug))]
    rng, diffs = np.random.RandomState(seed), {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([by_group[i] for i in rng.randint(0, len(ug), len(ug))])
        if y[idx].min() == y[idx].max():
            continue
        a, b = metric_set(y[idx], p_cond[idx], thr), metric_set(y[idx], p_ctrl[idx], thr)
        for k in keys:
            diffs[k].append(a[k] - b[k])
    return {k: tuple(np.nanpercentile(v, [2.5, 97.5])) if len(v) >= 0.8 * n_boot else (float("nan"),) * 2
            for k, v in diffs.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description="Validation-only perturbation audit for DenseNet121 seeds 42-44.")
    p.add_argument("--data-root")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--audit-dir", default=str(DEFAULT_OUTPUT_ROOT / "shortcut_audit"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "perturbation_audit_densenet"))
    p.add_argument("--batch-size", type=int, default=32, help="keep 32 so control matches the saved predictions")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args(argv)

    set_seed(args.seed)
    root = resolve_data_root(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    amp = device.type == "cuda"
    print("Validation split only; the test split is not read. Inference only; no training.")

    # ---- alignment: manifest <-> metadata, and original sizes from the images themselves
    meta = pd.read_csv(Path(args.audit_dir) / "metadata_val.csv")
    train_meta = pd.read_csv(Path(args.audit_dir) / "metadata_train.csv")
    frame = load_split(args.manifest, root, "val")
    allm.check_alignment(meta, train_meta, frame)
    order = frame.rel_path.tolist()
    meta = meta.set_index("path").loc[order].reset_index()
    targets = meta.target.values
    grays = [load_gray(pth) for pth in frame.abs_path]
    sizes = np.array([g.size for g in grays])
    if not ((sizes[:, 0] == meta.width.values) & (sizes[:, 1] == meta.height.values)).all():
        raise RuntimeError("image sizes read from disk disagree with metadata_val.csv")
    print(f"Alignment OK: {len(frame)} val images match the manifest by path, label and patient_id; "
          "sizes match metadata.")
    pad_masks = [region_masks(int(w), int(h))[1] for w, h in sizes]
    pad_frac = np.array([m.mean() for m in pad_masks])

    # ---- models + reference (saved-pipeline) predictions
    models, thr, ref = {}, {}, {}
    ds_ref = None
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
        loader, fr = make_loader(args.manifest, root, "val", args.batch_size, num_workers=0, seed=args.seed,
                                 train=False, pin_memory=device.type == "cuda", preprocess=pre)
        if fr.rel_path.tolist() != order:
            raise RuntimeError(f"{run}: loader order differs from metadata order")
        ds_ref = loader.dataset
        pr, tg, _ = predict(model, loader, device, amp=amp, desc=f"s{seed} reference")
        if not (tg == targets).all():
            raise RuntimeError(f"{run}: label order mismatch")
        saved = load_json(run_dir / "val_metrics.json")["roc_auc"]
        if abs(roc_auc_score(tg, pr) - saved) > AUC_TOL:
            raise RuntimeError(f"{run}: reference AUC does not reproduce saved {saved}")
        models[seed], thr[seed], ref[seed] = model, load_json(run_dir / "threshold.json")["threshold"], pr
        print(f"seed {seed} ({run}): densenet121/{pre}; reference pooled AUC {roc_auc_score(tg, pr):.6f} "
              f"vs saved {saved:.6f} OK; saved threshold {thr[seed]:.4f}")

    # ---- build each condition's tensors once; verify control == saved pipeline; verify pad geometry
    probs, changed_area = {}, {}
    ctrl_stack = None
    align = []
    tgt_t = torch.tensor(targets, dtype=torch.long)
    black = lambda v: torch.tensor([(v / 255.0 - m) / s for m, s in zip(MEAN, STD)]).view(3, 1)  # fill 0-255 -> normalised
    for cond in CONDITIONS:
        stack = torch.stack([to_tensor(g, cond) for g in tqdm(grays, desc=f"build {cond}", leave=False)])
        if cond == "control":
            ctrl_stack = stack
            ref_t = torch.stack([ds_ref[i][0] for i in range(len(ds_ref))])
            if not torch.equal(stack, ref_t):
                raise RuntimeError("control tensors differ from the saved dataset pipeline's tensors")
            del ref_t
            print("control tensors are bit-identical to the saved pipeline's tensors for all 878 images.")
        # padding region must hold exactly the fill value (verifies identified geometry on every image)
        if cond in ("control", "pad_imgmean", "pad_gray128", "pad_margin10_imgmean"):
            bad = 0
            for i, g in enumerate(grays):
                pm = torch.from_numpy(pad_masks[i])
                if not pm.any():
                    continue
                fill = {"control": 0, "pad_gray128": 128}.get(cond, int(round(np.asarray(g).mean())))
                if not torch.allclose(stack[i][:, pm], black(fill).expand(3, int(pm.sum())), atol=TENSOR_TOL):
                    bad += 1
            if bad:
                raise RuntimeError(f"{cond}: padding region mismatch on {bad} images")
        diff = (stack - ctrl_stack).abs().amax(dim=1) > TENSOR_TOL  # (N,H,W)
        changed_area[cond] = float(diff.float().mean(dim=(1, 2)).mean())
        loader = DataLoader(TensorDataset(stack, tgt_t), batch_size=args.batch_size, shuffle=False)
        for seed in RUNS:
            pr, tg, _ = predict(models[seed], loader, device, amp=amp, desc=f"s{seed} {cond}", criterion=None)
            if not (tg == targets).all():
                raise RuntimeError("target order changed")
            probs[(seed, cond)] = pr
        if cond != "control":
            del stack
    for seed in RUNS:
        d = np.abs(probs[(seed, "control")] - ref[seed]).max()
        align.append({"seed": seed, "control_vs_saved_pipeline_max_abs_prob_diff": float(d),
                      "saved_threshold": thr[seed]})
        if d > 1e-4:
            raise RuntimeError(f"seed {seed}: control predictions differ from saved-pipeline predictions (max {d})")
        print(f"seed {seed}: control predictions match the saved pipeline (max |dp| = {d:.2e})")
    pd.DataFrame(align).to_csv(out / "alignment_checks.csv", index=False)
    print("\nMean fraction of the 224x224 canvas whose normalised value differs from control (per condition):")
    print({c: round(v, 3) for c, v in changed_area.items()})

    # ---- metrics per (seed, condition) and paired differences from control
    rows, drows, crow = [], [], []
    y, patient = targets, meta.patient_id.values
    for seed in RUNS:
        pc = probs[(seed, "control")]
        for cond in CONDITIONS:
            pr = probs[(seed, cond)]
            m = metric_set(y, pr, thr[seed])
            changed = (pr >= thr[seed]) != (pc >= thr[seed])
            m.update({"seed": seed, "condition": cond, "threshold": thr[seed], "changed_area_frac": changed_area[cond],
                      "pred_changed": int(changed.sum()), "mean_abs_dprob": float(np.abs(pr - pc).mean()),
                      "new_false_positives": int((changed & (pr >= thr[seed]) & (y == 0)).sum()),
                      "new_false_negatives": int((changed & (pr < thr[seed]) & (y == 1)).sum()),
                      "fixed_errors": int((changed & (((pr >= thr[seed]) == (y == 1)))).sum())})
            rows.append(m)
            for lab, name in ((0, "NORMAL"), (1, "PNEUMONIA")):
                s = y == lab
                crow.append({"seed": seed, "condition": cond, "true_label": name, "n": int(s.sum()),
                             "mean_prob": float(pr[s].mean()), "mean_prob_control": float(pc[s].mean()),
                             "mean_abs_dprob": float(np.abs(pr[s] - pc[s]).mean()), "pred_changed": int(changed[s].sum()),
                             "correct_rate_at_saved_thr": float(((pr[s] >= thr[seed]) == (lab == 1)).mean())})
            if cond != "control":
                ci = paired_bootstrap(y, patient, pr, pc, thr[seed], args.n_boot, args.seed)
                a, b = metric_set(y, pr, thr[seed]), metric_set(y, pc, thr[seed])
                for k, (lo, hi) in ci.items():
                    drows.append({"seed": seed, "condition": cond, "metric": k, "control": b[k], "perturbed": a[k],
                                  "delta": a[k] - b[k], "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics_by_seed_condition.csv", index=False)
    dd = pd.DataFrame(drows)
    dd.to_csv(out / "paired_differences_vs_control.csv", index=False)
    pd.DataFrame(crow).to_csv(out / "classwise_by_seed_condition.csv", index=False)
    long = pd.concat([pd.DataFrame({"seed": s, "condition": c, "val_index": np.arange(len(y)), "path": order,
                                    "label": meta.label.values, "prob": probs[(s, c)],
                                    "prob_control": probs[(s, "control")], "threshold": thr[s]})
                      for s in RUNS for c in CONDITIONS])
    long.to_csv(out / "per_image_predictions.csv", index=False)

    show = ["auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "accuracy", "pred_changed",
            "mean_abs_dprob", "new_false_positives", "new_false_negatives"]
    for seed in RUNS:
        print(f"\n=== seed {seed} (threshold {thr[seed]:.4f}, fixed) ===")
        print(res[res.seed == seed].set_index("condition")[show].round(4).to_string())
    agg = res.groupby("condition", sort=False)[show].agg(["mean", "std"]).round(4)
    agg.to_csv(out / "mean_sd_across_seeds.csv")
    print("\nMean +- sample SD across the 3 seeds:")
    for c in CONDITIONS:
        s = res[res.condition == c]
        print(f"  {c:22s} " + "; ".join(f"{k} {s[k].mean():.4f}+-{s[k].std():.4f}" for k in
                                         ("auc", "pr_auc", "sensitivity", "specificity", "f1", "accuracy", "mean_abs_dprob")) +
              f"; changed {s.pred_changed.mean():.1f}")
    print("\nPaired differences from control, with 95% patient-bootstrap CI (delta [lo, hi]):")
    for metric in ("auc", "sensitivity", "specificity"):
        sub = dd[dd.metric == metric]
        piv = {(r.condition, r.seed): f"{r.delta:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}]" for r in sub.itertuples()}
        print(f"  -- {metric}")
        for c in CONDITIONS[1:]:
            print(f"    {c:22s} " + " | ".join(f"s{s}: {piv[(c, s)]}" for s in RUNS))
    cw = pd.DataFrame(crow)
    print("\nClass-wise (mean |dprob|, predictions changed) by condition, mean over seeds:")
    print(cw[cw.condition != "control"].groupby(["condition", "true_label"], sort=False)[["mean_abs_dprob", "pred_changed"]].mean().round(4).to_string())

    # ---- does the effect scale with how much padding an image has? (pad conditions, terciles of pad area)
    ter = pd.qcut(pd.Series(pad_frac).rank(method="first"), 3, labels=["low pad", "mid pad", "high pad"])
    trows = []
    for seed in RUNS:
        for cond in ("pad_imgmean", "margin10_imgmean", "pad_margin10_imgmean"):
            d = np.abs(probs[(seed, cond)] - probs[(seed, "control")])
            ch = (probs[(seed, cond)] >= thr[seed]) != (probs[(seed, "control")] >= thr[seed])
            for t in ter.cat.categories:
                s = (ter == t).values
                trows.append({"seed": seed, "condition": cond, "pad_tercile": t, "n": int(s.sum()),
                              "pad_frac_median": float(np.median(pad_frac[s])), "mean_abs_dprob": float(d[s].mean()),
                              "pred_changed": int(ch[s].sum()),
                              "pneumonia_share": float(y[s].mean())})
    tt = pd.DataFrame(trows)
    tt.to_csv(out / "by_padding_amount_tercile.csv", index=False)
    print("\nMean |dprob| / changed predictions by amount of padding (terciles), mean over seeds:")
    print(tt.groupby(["condition", "pad_tercile"], sort=False)[["pad_frac_median", "mean_abs_dprob", "pred_changed", "pneumonia_share"]].mean().round(4).to_string())

    # ---- plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {42: "#2a9d6f", 43: "#d1603d", 44: "#3b6fb6"}
    for ax, (metric, lab) in zip(axes, (("auc", "ROC-AUC"), ("sensitivity", "Sensitivity (saved thr)"), ("specificity", "Specificity (saved thr)"))):
        for seed in RUNS:
            s = res[res.seed == seed]
            ax.plot(range(len(CONDITIONS)), s[metric].values, "o-", color=cols[seed], label=f"seed {seed}")
        ax.set_xticks(range(len(CONDITIONS)), CONDITIONS, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("DenseNet121, validation: metrics under perturbation (thresholds fixed at saved values). Exploratory.", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "metrics_by_condition.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, ("auc", "sensitivity", "specificity")):
        for k, seed in enumerate(RUNS):
            s = dd[(dd.seed == seed) & (dd.metric == metric)].set_index("condition").loc[CONDITIONS[1:]]
            x = np.arange(len(s)) + (k - 1) * 0.22
            ax.errorbar(x, s.delta, yerr=[s.delta - s.ci_lo, s.ci_hi - s.delta], fmt="o", capsize=3, color=cols[seed], label=f"seed {seed}")
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set_xticks(range(len(CONDITIONS) - 1), CONDITIONS[1:], rotation=40, ha="right", fontsize=7)
        ax.set_title(f"change in {metric} vs control (95% patient-bootstrap CI)", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "paired_differences.png", dpi=140)
    plt.close(fig)

    rng = np.random.RandomState(args.seed)
    ex = sorted(rng.choice(np.flatnonzero(y == 0), 2, replace=False).tolist() + rng.choice(np.flatnonzero(y == 1), 2, replace=False).tolist())
    fig, axes = plt.subplots(len(ex), len(CONDITIONS), figsize=(2.1 * len(CONDITIONS), 2.2 * len(ex)), squeeze=False)
    for r_, i in enumerate(ex):
        for c_, cond in enumerate(CONDITIONS):
            axes[r_][c_].imshow(denorm(to_tensor(grays[i], cond)), cmap="gray", vmin=0, vmax=1)
            axes[r_][c_].set_xticks([]); axes[r_][c_].set_yticks([])
            axes[r_][c_].set_title(cond if r_ == 0 else "", fontsize=6)
        axes[r_][0].set_ylabel(f"val{i:03d}\n{meta.label[i][:4]}", fontsize=7)
    fig.suptitle("Model inputs under each condition (seeded sample of validation images)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "perturbation_examples.png", dpi=130)
    plt.close(fig)

    save_json({"note": "Exploratory; validation only; inference only; test split never read. Changes are not proof of "
                       "shortcut use; masks can create OOD inputs and remove anatomy. Markers not masked.",
               "conditions": CONDITIONS, "changed_area_frac": changed_area, "alignment": align,
               "thresholds": {str(s): thr[s] for s in RUNS}, "n_boot": args.n_boot,
               "pad_area_frac": {"min": float(pad_frac.min()), "median": float(np.median(pad_frac)),
                                 "max": float(pad_frac.max())}}, out / "summary.json")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
