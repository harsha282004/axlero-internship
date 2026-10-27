"""Size-matched stress test across ALL six baseline checkpoints (validation only, inference only).

Wrapper around scripts/size_matched_stress_test.py: it imports that script's statistics functions
(auc, patient-clustered bootstrap CI, bin construction, caliper matching, per-subset evaluation)
unchanged and adds the multi-run loop, alignment checks, aggregation and plots.

  * Bins (pixel count, height) and the matched subset are built ONCE from validation metadata,
    independent of any model, and are identical for every checkpoint.
  * Each checkpoint is run with its own saved preprocessing.
  * Bootstrap resamples use the same RNG seed for every series, so CIs are paired across runs.
  * The test split is never read. Nothing is trained.

Run from project-1-healthcare/:
    .venv\\Scripts\\python.exe scripts\\size_matched_stress_test_all.py --data-root C:\\Users\\harsh\\dataset\\kermany
Outputs: runs/size_matched_stress_test_6runs/ (git-ignored).
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

import size_matched_stress_test as base  # existing script, imported unchanged
from shortcut_audit import FEATURES, models as metadata_models
from src.dataset import load_split, make_loader
from src.evaluate import load_model
from src.train import predict
from src.utils import (DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, load_json, resolve_data_root,
                       save_json, set_seed)

RUN_DIRS = ["baseline_aspect_20ep", "baseline_resize_20ep", "baseline_pad_seed43",
            "baseline_resize_seed43", "baseline_pad_seed44", "baseline_resize_seed44"]
MODE_COLOR = {"pad": "#d1603d", "resize": "#3b6fb6", "metadata": "#6b7a8f"}
SEED_MARKER = {42: "o", 43: "s", 44: "^"}
AUC_TOL = 1e-4


def run_label(mode, seed):
    return f"{mode}_s{seed}"


def check_alignment(meta, train_meta, frame):
    if set(meta.path) != set(frame.rel_path) or not meta.path.is_unique:
        raise RuntimeError("metadata_val.csv paths do not match the manifest val split 1:1")
    j = frame[["rel_path", "label", "patient_id"]].merge(meta[["path", "label", "patient_id"]],
                                                         left_on="rel_path", right_on="path")
    if len(j) != len(frame) or (j.label_x != j.label_y).any() or (j.patient_id_x != j.patient_id_y).any():
        raise RuntimeError("label/patient_id mismatch between manifest and metadata_val.csv")
    if set(train_meta.path) & set(meta.path) or set(train_meta.patient_id) & set(meta.patient_id):
        raise RuntimeError("train metadata overlaps validation (path or patient)")


def rows_for_subset(df, subset_name, by, bin_name, probs, thresholds, info, min_per_class, n_boot, seed):
    """Long-format rows, one per series. Fresh identical RNG per series => paired bootstrap resamples."""
    out = []
    for name, p in probs.items():
        r = base.evaluate_subset(df, {name: p}, thresholds, min_per_class, n_boot, np.random.RandomState(seed))
        out.append({"subset": subset_name, "bin_by": by, "bin": bin_name, "series": name, **info[name],
                    "n": r["n"], "n_normal": r["n_normal"], "n_pneumonia": r["n_pneumonia"],
                    "eligible": r["eligible"],
                    "status": "ok" if r["eligible"] else f"AUC undefined: fewer than {min_per_class} per class",
                    "auc": r[f"auc_{name}"], "ci_lo": r[f"auc_{name}_lo"], "ci_hi": r[f"auc_{name}_hi"],
                    "sens_at_saved_thr": r.get(f"sens_{name}", float("nan")),
                    "spec_at_saved_thr": r.get(f"spec_{name}", float("nan"))})
    return out


def plot_bins(long_df, by, out_path):
    tab = long_df[(long_df.subset == "bin") & (long_df.bin_by == by)]
    bins = list(dict.fromkeys(tab.bin))
    series = list(dict.fromkeys(tab.series))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    width = 0.8 / len(series)
    for k, s in enumerate(series):
        t = tab[tab.series == s].set_index("bin").loc[bins]
        x = np.arange(len(bins)) - 0.4 + width * (k + 0.5)
        y, lo, hi = t.auc.values, t.ci_lo.values, t.ci_hi.values
        mode, seed = t["mode"].iloc[0], t["seed"].iloc[0]
        ax.errorbar(x, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], fmt=SEED_MARKER.get(seed, "x"),
                    ms=5, capsize=2, color=MODE_COLOR[mode], mfc=MODE_COLOR[mode] if mode != "metadata" else "none",
                    label=s)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    n = tab.drop_duplicates("bin").set_index("bin").loc[bins]
    ax.set_xticks(np.arange(len(bins)), [f"{b}\nN={a}/P={c}" for b, a, c in zip(bins, n.n_normal, n.n_pneumonia)],
                  fontsize=6.5)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("ROC-AUC within bin (95% patient-bootstrap CI)")
    ax.set_title(f"Validation, bins by {by}: six checkpoints vs metadata-only (blank = AUC undefined)", fontsize=10)
    ax.legend(ncol=4, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_matched(m_df, out_path):
    m = m_df.copy()
    order = [s for s in m.series]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(m))
    y, lo, hi = m.auc.values, m.ci_lo.values, m.ci_hi.values
    for xi, (_, r) in zip(x, m.iterrows()):
        ax.errorbar(xi, r.auc, yerr=[[max(r.auc - r.ci_lo, 0)], [max(r.ci_hi - r.auc, 0)]],
                    fmt=SEED_MARKER.get(r.seed, "x"), ms=6, capsize=3, color=MODE_COLOR[r["mode"]],
                    mfc=MODE_COLOR[r["mode"]] if r["mode"] != "metadata" else "none")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x, order, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("ROC-AUC (95% patient-bootstrap CI)")
    ax.set_title("Size-matched subset (same pairs for all runs); validation only; exploratory", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Validation-only size-matched stress test over six checkpoints.")
    p.add_argument("--data-root")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--audit-dir", default=str(DEFAULT_OUTPUT_ROOT / "shortcut_audit"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "size_matched_stress_test_6runs"))
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--min-per-class", type=int, default=10)
    p.add_argument("--caliper", type=float, default=0.10)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42, help="seed for matching/bootstrap (not model seeds)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args(argv)

    set_seed(args.seed)
    root = resolve_data_root(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print("Validation split only; the test split is not read. Exploratory; inference only.")

    # ---- alignment of manifest <-> metadata
    meta = pd.read_csv(Path(args.audit_dir) / "metadata_val.csv")
    train_meta = pd.read_csv(Path(args.audit_dir) / "metadata_train.csv")
    frame = load_split(args.manifest, root, "val")
    check_alignment(meta, train_meta, frame)
    order = frame.rel_path.tolist()
    meta = meta.set_index("path").loc[order].reset_index()  # loader order
    print(f"Alignment OK: {len(frame)} val images match the manifest by path, label and patient_id.")

    # ---- inference for each checkpoint with its own saved preprocessing
    probs, thresholds, info, align_rows = {}, {}, {}, []
    for run in RUN_DIRS:
        run_dir = DEFAULT_OUTPUT_ROOT / run
        cfg = load_json(run_dir / "config.json")
        model, ckpt = load_model(run_dir / "best.pt", device)
        pre_ckpt = ckpt["model"].get("preprocess", "resize")
        if pre_ckpt != cfg["preprocess"] or ckpt.get("seed") != cfg["seed"]:
            raise RuntimeError(f"{run}: checkpoint ({pre_ckpt}, seed {ckpt.get('seed')}) disagrees with "
                               f"config.json ({cfg['preprocess']}, seed {cfg['seed']})")
        label = run_label(pre_ckpt, cfg["seed"])
        loader, fr = make_loader(args.manifest, root, "val", 64, num_workers=0, seed=args.seed, train=False,
                                 pin_memory=device.type == "cuda", preprocess=pre_ckpt)
        if fr.rel_path.tolist() != order:
            raise RuntimeError(f"{run}: loader order differs from metadata order")
        pr, targets, _ = predict(model, loader, device, amp=device.type == "cuda", desc=label)
        if not (targets == meta.target.values).all():
            raise RuntimeError(f"{run}: label order mismatch")
        pooled, saved = base.auc_or_nan(targets, pr), load_json(run_dir / "val_metrics.json")["roc_auc"]
        if abs(pooled - saved) > AUC_TOL:
            raise RuntimeError(f"{run}: pooled val AUC {pooled:.6f} does not reproduce saved {saved:.6f}")
        probs[label], thresholds[label] = pr, load_json(run_dir / "threshold.json")["threshold"]
        info[label] = {"run": run, "mode": pre_ckpt, "seed": cfg["seed"]}
        align_rows.append({"run": run, "series": label, "preprocess": pre_ckpt, "seed": cfg["seed"],
                           "pooled_auc_here": pooled, "pooled_auc_saved": saved, "abs_diff": abs(pooled - saved),
                           "threshold": thresholds[label]})
        print(f"{label:12s} ({run}): pooled val AUC {pooled:.6f} vs saved {saved:.6f}  OK")

    lr = metadata_models(args.seed)["logistic_regression"].fit(train_meta[FEATURES].values, train_meta.target.values)
    probs["metadata_lr"] = lr.predict_proba(meta[FEATURES].values)[:, 1]
    info["metadata_lr"] = {"run": "metadata_lr(fit on train)", "mode": "metadata", "seed": -1}
    align_rows.append({"run": "metadata_lr", "series": "metadata_lr", "preprocess": "n/a", "seed": -1,
                       "pooled_auc_here": base.auc_or_nan(meta.target.values, probs["metadata_lr"]),
                       "pooled_auc_saved": float("nan"), "abs_diff": float("nan"), "threshold": float("nan")})
    pd.DataFrame(align_rows).to_csv(out / "alignment_checks.csv", index=False)
    pd.DataFrame({"path": meta.path, "patient_id": meta.patient_id, "label": meta.label, "height": meta.height,
                  "pixels": meta.pixels, **{f"prob_{k}": v for k, v in probs.items()}}
                 ).to_csv(out / "val_predictions_all_runs.csv", index=False)

    # ---- bins (built once from metadata) -> long table
    long_rows = []
    for by in ("pixels", "height"):
        binned = base.make_bins(meta, by, args.n_bins)
        for b in binned.cat.categories:
            m = (binned == b).values
            long_rows += rows_for_subset(meta[m], "bin", by, f"{b.left:.0f}-{b.right:.0f}",
                                         {k: v[m] for k, v in probs.items()}, thresholds, info,
                                         args.min_per_class, args.n_boot, args.seed)

    # ---- matched subset (built once from metadata) -> same pairs for every run
    matched, n_pairs = base.caliper_match(meta, np.random.RandomState(args.seed), args.caliper)
    idx = meta.index.get_indexer(matched.index)
    long_rows += rows_for_subset(matched, "matched", "log_pixels", f"caliper{args.caliper}",
                                 {k: v[idx] for k, v in probs.items()}, thresholds, info, 1, args.n_boot, args.seed)
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out / "results_long_all_runs.csv", index=False)

    # composition of the matched subset
    pos_all, neg_all = int((meta.target == 1).sum()), int((meta.target == 0).sum())
    excl = meta.drop(matched.index)
    comp = {"pairs": n_pairs, "included_images": 2 * n_pairs, "excluded_images": len(meta) - 2 * n_pairs,
            "pneumonia_total": pos_all, "pneumonia_excluded": pos_all - n_pairs,
            "normal_total": neg_all, "normal_excluded": neg_all - n_pairs, "caliper_log_pixels": args.caliper,
            "median_pixels_included": {c: float(matched[matched.label == c].pixels.median()) for c in ("NORMAL", "PNEUMONIA")},
            "median_pixels_excluded": {c: float(excl[excl.label == c].pixels.median()) for c in ("NORMAL", "PNEUMONIA")},
            "median_aspect_included": {c: float(matched[matched.label == c].aspect.median()) for c in ("NORMAL", "PNEUMONIA")}}
    print("\nMatched subset composition:", comp)

    # ---- wide AUC tables and per-mode summaries
    ok = long_df.copy()
    for by in ("pixels", "height"):
        t = ok[(ok.subset == "bin") & (ok.bin_by == by)]
        wide = t.pivot(index="bin", columns="series", values="auc").reindex(list(dict.fromkeys(t.bin)))
        counts = t.drop_duplicates("bin").set_index("bin")[["n", "n_normal", "n_pneumonia", "eligible"]]
        wide = counts.join(wide)
        wide.to_csv(out / f"bins_by_{by}_auc_wide.csv")
        print(f"\nBins by {by}: ROC-AUC per series (NaN = undefined, fewer than {args.min_per_class}/class)")
        print(wide.round(4).to_string())

    cnn = long_df[long_df["mode"] != "metadata"]
    summ = (cnn[cnn.eligible].groupby(["subset", "bin_by", "bin", "mode"]).auc.agg(["mean", "std", "min", "max", "count"])
            .round(4).reset_index())
    summ.to_csv(out / "per_mode_summary_by_bin.csv", index=False)
    print("\nPer-preprocessing mean/SD/min/max of within-bin AUC across the 3 seeds (eligible bins only):")
    print(summ.to_string(index=False))

    m_df = long_df[long_df.subset == "matched"].copy()
    m_df.to_csv(out / "matched_subset_all_runs.csv", index=False)
    print("\nMatched subset, per run:")
    print(m_df[["series", "n", "auc", "ci_lo", "ci_hi", "sens_at_saved_thr", "spec_at_saved_thr"]].round(4).to_string(index=False))
    mm = m_df[m_df["mode"] != "metadata"].groupby("mode")[["auc", "sens_at_saved_thr", "spec_at_saved_thr"]].agg(["mean", "std"]).round(4)
    mm.to_csv(out / "matched_subset_per_mode.csv")
    print("\nMatched subset, mean/SD across seeds per mode:")
    print(mm.to_string())

    for by in ("pixels", "height"):
        plot_bins(long_df, by, out / f"auc_by_{by}_bin_all_runs.png")
    plot_matched(m_df, out / "matched_subset_auc_all_runs.png")
    save_json({"note": "Exploratory; validation only; inference only; test split never read. Size matching does not "
                       "prove the absence of shortcuts or clinical generalisation.",
               "alignment": align_rows, "matched_composition": comp, "min_per_class": args.min_per_class,
               "n_bins": args.n_bins, "n_boot": args.n_boot}, out / "summary.json")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
