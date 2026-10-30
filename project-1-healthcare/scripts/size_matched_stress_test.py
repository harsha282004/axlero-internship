"""Size-matched stress test on the VALIDATION split (inference only, no training).

Question: do trained CNNs still separate NORMAL from PNEUMONIA among images of similar
size? ROC-AUC is computed within image-size bins and on a size-matched subset, for each
checkpoint (using the exact preprocessing it was trained with) and for a metadata-only
logistic regression / random forest fitted on TRAIN metadata (from the shortcut audit).

Only the val split is read for inference; the test split is never read. Size matching
does NOT prove the models are free of shortcuts (see the printed limitations).

Run from project-1-healthcare/:
    .venv\\Scripts\\python.exe scripts\\size_matched_stress_test.py --data-root C:\\Users\\harsh\\dataset\\kermany
Outputs: runs/size_matched_stress_test/ (git-ignored).
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
from sklearn.metrics import roc_auc_score

from shortcut_audit import FEATURES, models as metadata_models
from src.dataset import load_split, make_loader
from src.evaluate import load_model
from src.train import predict
from src.utils import (DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, load_json, resolve_data_root,
                       save_json, set_seed)

RUNS = {"pad_20ep": "baseline_aspect_20ep", "resize_20ep": "baseline_resize_20ep"}
NAN = float("nan")


def auc_or_nan(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else NAN


def cluster_bootstrap_ci(y, p, groups, n_boot, rng):
    """95% percentile CI of ROC-AUC, resampling whole patients."""
    ug, inv = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inv == i) for i in range(len(ug))]
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate([by_group[i] for i in rng.randint(0, len(ug), len(ug))])
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], p[idx]))
    if len(aucs) < 0.8 * n_boot:
        return NAN, NAN
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def sens_spec(y, p, thr):
    pred = p >= thr
    pos, neg = (y == 1).sum(), (y == 0).sum()
    return (float((pred & (y == 1)).sum() / pos) if pos else NAN,
            float((~pred & (y == 0)).sum() / neg) if neg else NAN)


def evaluate_subset(df, scores, thresholds, min_per_class, n_boot, rng):
    """One row of stats for the rows in df (scores: dict name -> prob array aligned to df)."""
    y = df.target.values
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    row = {"n": len(df), "n_normal": n0, "n_pneumonia": n1}
    eligible = min(n0, n1) >= min_per_class
    row["eligible"] = eligible
    for name, p in scores.items():
        if not eligible:
            row[f"auc_{name}"], row[f"auc_{name}_lo"], row[f"auc_{name}_hi"] = NAN, NAN, NAN
            continue
        row[f"auc_{name}"] = auc_or_nan(y, p)
        row[f"auc_{name}_lo"], row[f"auc_{name}_hi"] = cluster_bootstrap_ci(y, p, df.patient_id.values, n_boot, rng)
        if name in thresholds:
            row[f"sens_{name}"], row[f"spec_{name}"] = sens_spec(y, p, thresholds[name])
    return row


def make_bins(meta, by, n_bins):
    col = {"pixels": "pixels", "height": "height"}[by]
    binned = pd.qcut(meta[col], q=n_bins, duplicates="drop")
    return binned


def caliper_match(meta, rng, caliper):
    """1:1 match each PNEUMONIA image to an unused NORMAL image with similar log pixel count."""
    pos = meta[meta.target == 1].sample(frac=1, random_state=int(rng.randint(1 << 30)))
    neg = meta[meta.target == 0]
    used, pairs = set(), []
    nlp = neg.log_pixels.values
    for i, r in pos.iterrows():
        d = np.abs(nlp - r.log_pixels)
        d[[j in used for j in neg.index]] = np.inf
        k = int(np.argmin(d))
        if d[k] <= caliper:
            used.add(neg.index[k])
            pairs.append((i, neg.index[k]))
    keep = [i for pr in pairs for i in pr]
    return meta.loc[keep], len(pairs)


def plot_bins(tables, out_path):
    fig, axes = plt.subplots(len(tables), 1, figsize=(11, 4.2 * len(tables)), squeeze=False)
    colors = {"pad_20ep": "#d1603d", "resize_20ep": "#3b6fb6", "metadata_lr": "#6b7a8f"}
    for ax, (by, tab) in zip(axes[:, 0], tables.items()):
        x = np.arange(len(tab))
        for off, name in zip((-0.2, 0, 0.2), colors):
            y = tab[f"auc_{name}"].values
            lo, hi = tab[f"auc_{name}_lo"].values, tab[f"auc_{name}_hi"].values
            ax.errorbar(x + off, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt="o", ms=5,
                        capsize=3, color=colors[name], label=name)
        ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xticks(x, [f"{b}\nN={n0}/P={n1}" for b, n0, n1 in zip(tab.bin, tab.n_normal, tab.n_pneumonia)],
                      fontsize=6.5)
        ax.set_ylim(0.3, 1.02)
        ax.set_ylabel("ROC-AUC within bin (95% patient-bootstrap CI)")
        ax.set_title(f"Validation, bins by {by} (empty = fewer than the minimum per class)")
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("Size-matched stress test (exploratory; validation only; not proof of absence of shortcuts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Validation-only size-matched stress test (inference only).")
    p.add_argument("--data-root")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--audit-dir", default=str(DEFAULT_OUTPUT_ROOT / "shortcut_audit"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "size_matched_stress_test"))
    p.add_argument("--n-bins", type=int, default=8, help="quantile bins of the pooled validation set")
    p.add_argument("--min-per-class", type=int, default=10, help="AUC reported only if both classes >= this")
    p.add_argument("--caliper", type=float, default=0.10, help="max |diff| in log pixel count for matched pairs")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args(argv)

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    root = resolve_data_root(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print("Validation split only; the test split is not read. Exploratory; inference only.")

    # ---- alignment: manifest val rows <-> saved metadata, by relative path (hard failures, no guessing)
    meta = pd.read_csv(Path(args.audit_dir) / "metadata_val.csv")
    train_meta = pd.read_csv(Path(args.audit_dir) / "metadata_train.csv")
    frame = load_split(args.manifest, root, "val")
    if set(meta.path) != set(frame.rel_path) or not meta.path.is_unique:
        raise RuntimeError("metadata_val.csv paths do not match the manifest val split 1:1")
    joined = frame[["rel_path", "label", "patient_id"]].merge(meta[["path", "label", "patient_id"]],
                                                              left_on="rel_path", right_on="path")
    if len(joined) != len(frame) or (joined.label_x != joined.label_y).any() \
            or (joined.patient_id_x != joined.patient_id_y).any():
        raise RuntimeError("label/patient_id mismatch between manifest and metadata_val.csv")
    if set(train_meta.path) & set(meta.path) or set(train_meta.patient_id) & set(meta.patient_id):
        raise RuntimeError("train metadata overlaps validation (path or patient)")
    print(f"Alignment OK: {len(frame)} val images match the manifest by path, label and patient_id.")

    # ---- CNN inference on val with each checkpoint's own preprocessing
    order = frame.rel_path.tolist()
    meta = meta.set_index("path").loc[order].reset_index()  # same order as the loader
    scores, thresholds, check = {}, {}, {}
    for key, run in RUNS.items():
        run_dir = DEFAULT_OUTPUT_ROOT / run
        model, ckpt = load_model(run_dir / "best.pt", device)
        preprocess = ckpt["model"].get("preprocess", "resize")
        thresholds[key] = load_json(run_dir / "threshold.json")["threshold"]
        loader, fr = make_loader(args.manifest, root, "val", 64, num_workers=0, seed=args.seed, train=False,
                                 pin_memory=device.type == "cuda", preprocess=preprocess)
        assert fr.rel_path.tolist() == order
        probs, targets, _ = predict(model, loader, device, amp=device.type == "cuda", desc=key)
        assert (targets == meta.target.values).all(), "label order mismatch"
        scores[key] = probs
        saved = load_json(run_dir / "val_metrics.json")["roc_auc"]
        check[key] = {"preprocess": preprocess, "pooled_auc_here": auc_or_nan(targets, probs),
                      "pooled_auc_saved": saved}
        print(f"{key}: preprocess={preprocess} thr={thresholds[key]:.4f} pooled val AUC here "
              f"{check[key]['pooled_auc_here']:.4f} vs saved {saved:.4f}")

    # ---- metadata-only baseline: fit on TRAIN metadata, predict VAL (no val labels used for fitting)
    lr = metadata_models(args.seed)["logistic_regression"].fit(train_meta[FEATURES].values, train_meta.target.values)
    scores["metadata_lr"] = lr.predict_proba(meta[FEATURES].values)[:, 1]
    print(f"metadata_lr pooled val AUC {auc_or_nan(meta.target.values, scores['metadata_lr']):.4f}")

    pd.DataFrame({"path": meta.path, "patient_id": meta.patient_id, "label": meta.label,
                  "height": meta.height, "pixels": meta.pixels,
                  **{f"prob_{k}": v for k, v in scores.items()}}).to_csv(out / "val_predictions.csv", index=False)

    # ---- bins
    tables = {}
    for by in ("pixels", "height"):
        bins = make_bins(meta, by, args.n_bins)
        rows = []
        for b in bins.cat.categories:
            m = (bins == b).values
            row = {"bin_by": by, "bin": f"{b.left:.0f}-{b.right:.0f}"}
            row.update(evaluate_subset(meta[m], {k: v[m] for k, v in scores.items()}, thresholds,
                                       args.min_per_class, args.n_boot, rng))
            rows.append(row)
        tab = pd.DataFrame(rows)
        tables[by] = tab
        tab.to_csv(out / f"bins_by_{by}.csv", index=False)
        cols = ["bin", "n_normal", "n_pneumonia"] + [f"auc_{k}" for k in scores]
        print(f"\nBins by {by} (AUC shown only when both classes >= {args.min_per_class}; NaN = unavailable):")
        print(tab[cols].round(4).to_string(index=False))

    # ---- caliper-matched subset
    matched, n_pairs = caliper_match(meta, rng, args.caliper)
    idx = meta.index.get_indexer(matched.index)
    mrow = evaluate_subset(matched, {k: v[idx] for k, v in scores.items()}, thresholds, 1, args.n_boot, rng)
    mrow.update({"pairs": n_pairs, "caliper_log_pixels": args.caliper,
                 "median_aspect_normal": float(matched[matched.target == 0].aspect.median()),
                 "median_aspect_pneumonia": float(matched[matched.target == 1].aspect.median()),
                 "median_pixels_normal": float(matched[matched.target == 0].pixels.median()),
                 "median_pixels_pneumonia": float(matched[matched.target == 1].pixels.median())})
    print(f"\nCaliper-matched subset ({n_pairs} 1:1 pairs of {int((meta.target == 1).sum())} PNEUMONIA possible; "
          f"|log-pixel diff| <= {args.caliper}):")
    print(pd.Series({k: (round(v, 4) if isinstance(v, float) else v) for k, v in mrow.items()}).to_string())

    plot_bins(tables, out / "auc_by_size_bin.png")
    save_json({"note": "Exploratory, validation only, inference only. Size matching does not prove absence of shortcuts.",
               "alignment_checks": check, "matched": mrow, "min_per_class": args.min_per_class,
               "n_bins": args.n_bins}, out / "summary.json")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
