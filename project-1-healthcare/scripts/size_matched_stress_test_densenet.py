"""Size-matched stress test for the DenseNet121 checkpoint (validation only, inference only).

Reuses scripts/size_matched_stress_test.py and scripts/size_matched_stress_test_all.py unchanged.
The same quantile bins and the same caliper-matched pairs as the six-checkpoint audit are
rebuilt from validation metadata and verified against that audit's saved outputs. The baseline
seed-42 predictions (pad and resize) are read from the audit's val_predictions_all_runs.csv, so
DenseNet121 and the baselines are compared on IDENTICAL images, bins and pairs.

The test split is never read. Nothing is trained. Size matching does not prove the absence of
shortcuts or clinical generalisation.

Run from project-1-healthcare/:
    .venv\\Scripts\\python.exe scripts\\size_matched_stress_test_densenet.py --data-root C:\\Users\\harsh\\dataset\\kermany
Outputs: runs/size_matched_stress_test_densenet/ (git-ignored).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import size_matched_stress_test as base               # existing, unchanged
import size_matched_stress_test_all as allm           # existing, unchanged
from shortcut_audit import FEATURES, models as metadata_models
from src.dataset import load_split, make_loader
from src.evaluate import load_model
from src.train import predict
from src.utils import (DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, load_json, resolve_data_root,
                       save_json, set_seed)

DENSE_RUN = "densenet121_pad_seed42"
DENSE = "densenet121_pad_s42"
BASELINES = {"pad_s42": "baseline_aspect_20ep", "resize_s42": "baseline_resize_20ep"}
AUDIT_DIR6 = DEFAULT_OUTPUT_ROOT / "size_matched_stress_test_6runs"
AUC_TOL = 1e-4


def paired_auc_diff(df, a, b, n_boot, seed):
    """AUC(a) - AUC(b) on the same rows with a patient-clustered paired bootstrap 95% CI."""
    y, g = df.target.values, df.patient_id.values
    ug, inv = np.unique(g, return_inverse=True)
    by_group = [np.flatnonzero(inv == i) for i in range(len(ug))]
    point = roc_auc_score(y, a) - roc_auc_score(y, b)
    rng, diffs = np.random.RandomState(seed), []
    for _ in range(n_boot):
        idx = np.concatenate([by_group[i] for i in rng.randint(0, len(ug), len(ug))])
        if y[idx].min() != y[idx].max():
            diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5]) if len(diffs) >= 0.8 * n_boot else (float("nan"),) * 2
    return float(point), float(lo), float(hi)


def main(argv=None):
    p = argparse.ArgumentParser(description="Validation-only size-matched stress test for DenseNet121.")
    p.add_argument("--data-root")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--audit-dir", default=str(DEFAULT_OUTPUT_ROOT / "shortcut_audit"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "size_matched_stress_test_densenet"))
    p.add_argument("--n-bins", type=int, default=8)
    p.add_argument("--min-per-class", type=int, default=10)
    p.add_argument("--caliper", type=float, default=0.10)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42, help="matching/bootstrap seed (same as the six-run audit)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args(argv)

    set_seed(args.seed)
    root = resolve_data_root(args.data_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    allm.MODE_COLOR["densenet"] = "#2a9d6f"  # runtime-only addition for plotting; no source file is edited
    print("Validation split only; the test split is not read. Exploratory; inference only.")

    # ---- alignment: manifest <-> metadata (existing check, unchanged)
    meta = pd.read_csv(Path(args.audit_dir) / "metadata_val.csv")
    train_meta = pd.read_csv(Path(args.audit_dir) / "metadata_train.csv")
    frame = load_split(args.manifest, root, "val")
    allm.check_alignment(meta, train_meta, frame)
    order = frame.rel_path.tolist()
    meta = meta.set_index("path").loc[order].reset_index()
    print(f"Alignment OK: {len(frame)} val images match the manifest by path, label and patient_id.")

    # ---- DenseNet checkpoint: config <-> checkpoint consistency, inference with its own preprocessing
    run_dir = DEFAULT_OUTPUT_ROOT / DENSE_RUN
    cfg = load_json(run_dir / "config.json")
    model, ckpt = load_model(run_dir / "best.pt", device)
    pre = ckpt["model"].get("preprocess", "resize")
    if (pre != cfg["preprocess"] or ckpt.get("seed") != cfg["seed"] or ckpt["model"]["name"] != cfg["model"]
            or cfg.get("pretrained")):
        raise RuntimeError(f"checkpoint ({ckpt['model']['name']}, {pre}, seed {ckpt.get('seed')}) disagrees with "
                           f"config.json ({cfg['model']}, {cfg['preprocess']}, seed {cfg['seed']}, "
                           f"pretrained={cfg.get('pretrained')})")
    loader, fr = make_loader(args.manifest, root, "val", args.batch_size, num_workers=0, seed=args.seed, train=False,
                             pin_memory=device.type == "cuda", preprocess=pre)
    if fr.rel_path.tolist() != order:
        raise RuntimeError("DenseNet loader order differs from metadata order")
    dprob, targets, _ = predict(model, loader, device, amp=device.type == "cuda", desc=DENSE)
    if not (targets == meta.target.values).all():
        raise RuntimeError("DenseNet label order mismatch")
    pooled, saved = base.auc_or_nan(targets, dprob), load_json(run_dir / "val_metrics.json")["roc_auc"]
    if abs(pooled - saved) > AUC_TOL:
        raise RuntimeError(f"DenseNet pooled val AUC {pooled:.6f} does not reproduce saved {saved:.6f}")
    print(f"{DENSE}: preprocess={pre}, seed={cfg['seed']}, pretrained={cfg['pretrained']}; pooled val AUC "
          f"{pooled:.6f} vs saved {saved:.6f}  OK")
    align = [{"series": DENSE, "run": DENSE_RUN, "pooled_auc_here": pooled, "pooled_auc_saved": saved}]

    # ---- baseline seed-42 predictions from the six-run audit, aligned by path (identical images)
    pred6 = pd.read_csv(AUDIT_DIR6 / "val_predictions_all_runs.csv").set_index("path")
    if set(pred6.index) != set(order):
        raise RuntimeError("six-run predictions cover a different set of validation images")
    pred6 = pred6.loc[order]
    if not (pred6.label.values == meta.label.values).all() or not (pred6.patient_id.values == meta.patient_id.values).all():
        raise RuntimeError("six-run predictions disagree with manifest labels/patient_ids")
    probs, thresholds, info = {DENSE: dprob}, {DENSE: load_json(run_dir / "threshold.json")["threshold"]}, {
        DENSE: {"run": DENSE_RUN, "mode": "densenet", "seed": cfg["seed"], "preprocess": pre}}
    for label, run in BASELINES.items():
        probs[label] = pred6[f"prob_{label}"].values
        thresholds[label] = load_json(DEFAULT_OUTPUT_ROOT / run / "threshold.json")["threshold"]
        mode = load_json(DEFAULT_OUTPUT_ROOT / run / "config.json")["preprocess"]
        info[label] = {"run": run, "mode": mode, "seed": 42, "preprocess": mode}
        pa, sv = base.auc_or_nan(meta.target.values, probs[label]), load_json(DEFAULT_OUTPUT_ROOT / run / "val_metrics.json")["roc_auc"]
        if abs(pa - sv) > AUC_TOL:
            raise RuntimeError(f"{label}: saved six-run predictions give pooled AUC {pa:.6f}, expected {sv:.6f}")
        align.append({"series": label, "run": run, "pooled_auc_here": pa, "pooled_auc_saved": sv})
        print(f"{label}: baseline predictions reused from six-run audit; pooled AUC {pa:.6f} vs saved {sv:.6f}  OK")
    lr = metadata_models(args.seed)["logistic_regression"].fit(train_meta[FEATURES].values, train_meta.target.values)
    probs["metadata_lr"] = lr.predict_proba(meta[FEATURES].values)[:, 1]
    info["metadata_lr"] = {"run": "metadata_lr(fit on train)", "mode": "metadata", "seed": -1, "preprocess": "n/a"}

    # ---- subsets: bins and matched pairs rebuilt from metadata, then verified against the six-run audit
    subsets = []  # (subset, by, bin_name, positional row indices)
    for by in ("pixels", "height"):
        binned = base.make_bins(meta, by, args.n_bins)
        for b in binned.cat.categories:
            subsets.append(("bin", by, f"{b.left:.0f}-{b.right:.0f}", np.flatnonzero((binned == b).values)))
    matched, n_pairs = base.caliper_match(meta, np.random.RandomState(args.seed), args.caliper)
    subsets.append(("matched", "log_pixels", f"caliper{args.caliper}", meta.index.get_indexer(matched.index)))

    long_rows = []
    for subset, by, bname, idx in subsets:
        long_rows += allm.rows_for_subset(meta.iloc[idx], subset, by, bname, {k: v[idx] for k, v in probs.items()},
                                          thresholds, info, args.min_per_class if subset == "bin" else 1,
                                          args.n_boot, args.seed)
    long_df = pd.DataFrame(long_rows)

    six = pd.read_csv(AUDIT_DIR6 / "results_long_all_runs.csv")
    for (subset, by, bname), g in long_df.groupby(["subset", "bin_by", "bin"], sort=False):
        s = six[(six.subset == subset) & (six.bin_by == by) & (six.bin == bname)]
        if s.empty or (s.n.iloc[0], s.n_normal.iloc[0], s.n_pneumonia.iloc[0]) != (g.n.iloc[0], g.n_normal.iloc[0], g.n_pneumonia.iloc[0]):
            raise RuntimeError(f"subset {subset}/{by}/{bname} differs from the six-run audit (bins or pairs not identical)")
        for label in BASELINES:  # reproduce the audit's baseline numbers exactly
            a = g[g.series == label].auc.iloc[0]
            b6 = s[s.series == label].auc.iloc[0]
            if not (np.isnan(a) and np.isnan(b6)) and abs(a - b6) > 1e-9:
                raise RuntimeError(f"{label} AUC in {subset}/{by}/{bname} ({a}) != six-run audit ({b6})")
    print("Bins and matched pairs verified identical to the six-run audit; baseline AUCs reproduced exactly.")

    long_df.to_csv(out / "results_long.csv", index=False)
    pd.DataFrame(align).to_csv(out / "alignment_checks.csv", index=False)
    pd.DataFrame({"path": meta.path, "patient_id": meta.patient_id, "label": meta.label, "height": meta.height,
                  "pixels": meta.pixels, **{f"prob_{k}": v for k, v in probs.items()}}
                 ).to_csv(out / "val_predictions_densenet_vs_baselines.csv", index=False)

    # ---- tables
    for by in ("pixels", "height"):
        t = long_df[(long_df.subset == "bin") & (long_df.bin_by == by)]
        wide = t.drop_duplicates("bin").set_index("bin")[["n", "n_normal", "n_pneumonia", "eligible"]].join(
            t.pivot(index="bin", columns="series", values="auc"))
        wide = wide.reindex(list(dict.fromkeys(t.bin)))
        wide.to_csv(out / f"bins_by_{by}_auc_wide.csv")
        print(f"\nBins by {by}: ROC-AUC (NaN = undefined, fewer than {args.min_per_class} per class)")
        print(wide.round(4).to_string())
        ci = t[t.eligible & (t.series == DENSE)][["bin", "auc", "ci_lo", "ci_hi"]]
        print("DenseNet121 95% patient-bootstrap CIs:")
        print(ci.round(4).to_string(index=False))

    m = long_df[long_df.subset == "matched"].copy()
    m.to_csv(out / "matched_subset.csv", index=False)
    print(f"\nMatched subset ({n_pairs} pairs = {2 * n_pairs} images; {len(meta) - 2 * n_pairs} of {len(meta)} images excluded):")
    print(m[["series", "n", "auc", "ci_lo", "ci_hi", "sens_at_saved_thr", "spec_at_saved_thr"]].round(4).to_string(index=False))

    # ---- paired differences DenseNet - baseline seed 42, identical rows, paired bootstrap
    drows = []
    for subset, by, bname, idx in subsets:
        sub = meta.iloc[idx]
        n0, n1 = int((sub.target == 0).sum()), int((sub.target == 1).sum())
        if min(n0, n1) < (args.min_per_class if subset == "bin" else 1):
            continue
        for label in BASELINES:
            d, lo, hi = paired_auc_diff(sub, probs[DENSE][idx], probs[label][idx], args.n_boot, args.seed)
            drows.append({"subset": subset, "bin_by": by, "bin": bname, "n_normal": n0, "n_pneumonia": n1,
                          "baseline": label, "auc_densenet_minus_baseline": d, "ci_lo": lo, "ci_hi": hi})
    ddf = pd.DataFrame(drows)
    ddf.to_csv(out / "paired_differences_vs_baseline_seed42.csv", index=False)
    print("\nPaired AUC difference, DenseNet121 minus baseline seed 42 (same images; patient-bootstrap 95% CI):")
    print(ddf.round(4).to_string(index=False))

    allm.plot_bins(long_df, "pixels", out / "auc_by_pixels_bin_densenet.png")
    allm.plot_bins(long_df, "height", out / "auc_by_height_bin_densenet.png")
    allm.plot_matched(m, out / "matched_subset_auc_densenet.png")
    save_json({"note": "Exploratory; validation only; inference only; test split never read. Size matching does not "
                       "prove the absence of shortcuts or clinical generalisation.",
               "alignment": align, "matched_pairs": n_pairs, "included_images": 2 * n_pairs,
               "excluded_images": len(meta) - 2 * n_pairs, "verified_identical_to_six_run_audit": True},
              out / "summary.json")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
