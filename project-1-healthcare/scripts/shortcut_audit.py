"""Shortcut audit: do image METADATA alone (size, aspect ratio, colour mode) predict the label?

EXPLORATORY analysis, not a performance estimate. Uses the train and val splits of
data/splits.csv only; the test split is never read. Nothing here trains or selects the CNN.

Run from project-1-healthcare/:
    .venv\\Scripts\\python.exe scripts\\shortcut_audit.py --data-root C:\\Users\\harsh\\dataset\\kermany

Outputs go to runs/shortcut_audit/ (git-ignored).
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import load_split
from src.utils import DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, resolve_data_root, save_json

FEATURES = ["log_width", "log_height", "aspect", "log_pixels", "is_rgb"]
COLORS = {"NORMAL": "#3b6fb6", "PNEUMONIA": "#d1603d"}


def read_metadata(frame):
    """Width/height/mode from image headers (no pixel decoding needed)."""
    rows = []
    for r in frame.itertuples():
        with Image.open(r.abs_path) as im:
            w, h, mode = im.width, im.height, im.mode
        rows.append({"path": r.rel_path, "patient_id": r.patient_id, "label": r.label, "target": r.target,
                     "width": w, "height": h, "aspect": w / h, "pixels": w * h, "mode": mode,
                     "is_rgb": int(mode != "L")})
    meta = pd.DataFrame(rows)
    meta["log_width"], meta["log_height"] = np.log(meta.width), np.log(meta.height)
    meta["log_pixels"] = np.log(meta.pixels)
    return meta


def feature_tables(meta):
    num = ["width", "height", "aspect", "pixels"]
    summary = meta.groupby("label")[num].describe().T.round(3)
    tests = []
    for f in num + ["is_rgb"]:
        a, b = meta[meta.target == 0][f], meta[meta.target == 1][f]
        auc = roc_auc_score(meta.target, meta[f])
        row = {"feature": f, "median_NORMAL": a.median(), "median_PNEUMONIA": b.median(),
               "single_feature_auc": auc, "auc_orientation_free": max(auc, 1 - auc)}
        if f == "is_rgb":
            table = [[int((a == 1).sum()), int((a == 0).sum())], [int((b == 1).sum()), int((b == 0).sum())]]
            row["p_value"], row["test"] = fisher_exact(table)[1], "Fisher exact"
            row["rgb_NORMAL"], row["rgb_PNEUMONIA"] = table[0][0], table[1][0]
        else:
            row["p_value"], row["test"] = mannwhitneyu(a, b).pvalue, "Mann-Whitney U"
        tests.append(row)
    return summary, pd.DataFrame(tests)


def models(seed):
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=seed),
    }


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def cross_validate_val(meta, folds, seed):
    """Exploratory: grouped (by patient) stratified CV on the validation metadata only."""
    X, y, groups = meta[FEATURES].values, meta.target.values, meta.patient_id.values
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    out = []
    for name in models(seed):
        aucs, bals, oof = [], [], np.zeros(len(y))
        for tr, te in cv.split(X, y, groups):
            m = models(seed)[name].fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            oof[te] = p
            aucs.append(safe_auc(y[te], p))
            bals.append(balanced_accuracy_score(y[te], p >= 0.5))
        out.append({"model": name, "protocol": f"{folds}-fold patient-grouped CV on VAL (exploratory)",
                    "roc_auc_mean": np.nanmean(aucs), "roc_auc_sd": np.nanstd(aucs),
                    "roc_auc_pooled": safe_auc(y, oof), "balanced_acc_mean": np.mean(bals),
                    "n": len(y)})
    return out


def train_to_val(train_meta, val_meta, seed):
    """Fit on TRAIN metadata, evaluate once on VAL metadata (threshold fixed at 0.5, nothing tuned)."""
    out = []
    for name in models(seed):
        m = models(seed)[name].fit(train_meta[FEATURES].values, train_meta.target.values)
        p = m.predict_proba(val_meta[FEATURES].values)[:, 1]
        y, pred = val_meta.target.values, p >= 0.5
        sens = float((pred & (y == 1)).sum() / (y == 1).sum())
        spec = float((~pred & (y == 0)).sum() / (y == 0).sum())
        out.append({"model": name, "protocol": "fit on TRAIN metadata -> evaluate on VAL (thr 0.5)",
                    "roc_auc": safe_auc(y, p), "balanced_acc": balanced_accuracy_score(y, pred),
                    "accuracy": float((pred == y).mean()), "sensitivity": sens, "specificity": spec,
                    "majority_class_accuracy": float(max(y.mean(), 1 - y.mean())), "n_train": len(train_meta),
                    "n_val": len(y)})
    return out


def make_plots(val_meta, tests, results, out_dir):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for a, (col, title, log) in zip(ax.ravel(), [("width", "Width (px)", False), ("height", "Height (px)", False),
                                                 ("aspect", "Aspect ratio (w/h)", False),
                                                 ("pixels", "Pixel count (log x)", True)]):
        data = {k: val_meta[val_meta.label == k][col] for k in COLORS}
        bins = (np.logspace(np.log10(val_meta[col].min()), np.log10(val_meta[col].max()), 40) if log
                else np.linspace(val_meta[col].min(), val_meta[col].max(), 40))
        for k, v in data.items():
            a.hist(v, bins=bins, alpha=0.6, color=COLORS[k], label=k, density=True)
        a.set_title(title)
        a.set_ylabel("density")
        if log:
            a.set_xscale("log")
        a.legend()
    fig.suptitle("Validation split: image metadata by class (exploratory)")
    fig.tight_layout()
    fig.savefig(out_dir / "metadata_distributions.png", dpi=150)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 5))
    for k, c in COLORS.items():
        s = val_meta[val_meta.label == k]
        a.scatter(s.width, s.height, s=8, alpha=0.5, color=c, label=k)
    a.set_xlabel("width (px)")
    a.set_ylabel("height (px)")
    a.set_title("Validation split: width vs height")
    a.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "width_vs_height.png", dpi=150)
    plt.close(fig)

    names = [f"{r.feature}\n(single feature)" for r in tests.itertuples()]
    vals = tests.auc_orientation_free.tolist()
    for r in results:
        if "roc_auc" in r:
            names.append(f"{r['model']}\n(train->val)")
            vals.append(r["roc_auc"])
    fig, a = plt.subplots(figsize=(9, 4.5))
    a.bar(range(len(vals)), vals, color="#6b7a8f")
    a.axhline(0.5, color="gray", ls="--", lw=1)
    a.set_xticks(range(len(vals)), names, fontsize=7)
    a.set_ylim(0.4, 1.0)
    a.set_ylabel("ROC-AUC (orientation-free for single features)")
    a.set_title("How well metadata alone separates the classes (validation; exploratory)")
    for i, v in enumerate(vals):
        a.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "metadata_auc.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Exploratory metadata shortcut audit (train/val only; test untouched).")
    p.add_argument("--data-root", help="folder that CONTAINS chest_xray/ (or set CXR_DATA_ROOT)")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "shortcut_audit"))
    p.add_argument("--cnn-val-metrics", default=str(DEFAULT_OUTPUT_ROOT / "baseline_smoke" / "val_metrics.json"),
                   help="optional val_metrics.json of a CNN run, printed for comparison")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    root = resolve_data_root(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("EXPLORATORY metadata audit. Splits read: train, val. Test split is NOT read.")

    val_meta = read_metadata(load_split(args.manifest, root, "val"))
    train_meta = read_metadata(load_split(args.manifest, root, "train"))
    val_meta.to_csv(out_dir / "metadata_val.csv", index=False)
    train_meta.to_csv(out_dir / "metadata_train.csv", index=False)
    print(f"val: {len(val_meta)} images | train: {len(train_meta)} images")

    summary, tests = feature_tables(val_meta)
    summary.to_csv(out_dir / "feature_summary_val.csv")
    tests.to_csv(out_dir / "feature_tests_val.csv", index=False)
    print("\nVALIDATION metadata, class differences:")
    print(tests.round(4).to_string(index=False))

    cv_results = cross_validate_val(val_meta, args.folds, args.seed)
    tv_results = train_to_val(train_meta, val_meta, args.seed)
    pd.DataFrame(cv_results).to_csv(out_dir / "classifiers_val_cv.csv", index=False)
    pd.DataFrame(tv_results).to_csv(out_dir / "classifiers_train_to_val.csv", index=False)
    print("\nMetadata-only classifiers, grouped CV on VAL (exploratory):")
    print(pd.DataFrame(cv_results).round(4).to_string(index=False))
    print("\nMetadata-only classifiers, fit on TRAIN -> evaluated on VAL:")
    print(pd.DataFrame(tv_results).round(4).to_string(index=False))

    cnn = Path(args.cnn_val_metrics)
    if cnn.is_file():
        m = json.loads(cnn.read_text())
        print(f"\nFor reference, CNN validation ROC-AUC from {cnn.name}: {m['roc_auc']:.4f}")

    make_plots(val_meta, tests, tv_results, out_dir)
    save_json({"note": "Exploratory; train/val only; test never read.", "cv_on_val": cv_results,
               "train_to_val": tv_results}, out_dir / "summary.json")
    print(f"\nOutputs written to {out_dir}")


if __name__ == "__main__":
    main()
