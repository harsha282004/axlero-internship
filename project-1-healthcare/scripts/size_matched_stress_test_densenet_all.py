"""Size-matched stress test for DenseNet121 seeds 42/43/44 (validation only, inference only).

Reuses scripts/size_matched_stress_test.py and scripts/size_matched_stress_test_all.py unchanged.
Bins (pixel count, height) and the caliper-matched pairs are rebuilt from validation metadata and
verified against the six-checkpoint audit's saved output, so all three DenseNet checkpoints and the six
baseline checkpoints are evaluated on IDENTICAL images, bins and pairs. Baseline predictions are read
from that audit's val_predictions_all_runs.csv (aligned by path, pooled AUC re-verified).

The test split is never read. Nothing is trained. Size matching does not prove the absence of
shortcuts or clinical generalisation.

Run from project-1-healthcare/:
    .venv\\Scripts\\python.exe scripts\\size_matched_stress_test_densenet_all.py --data-root C:\\Users\\harsh\\dataset\\kermany
Outputs: runs/size_matched_stress_test_densenet_all/ (git-ignored).
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

import size_matched_stress_test as base               # existing, unchanged
import size_matched_stress_test_all as allm           # existing, unchanged
from shortcut_audit import FEATURES, models as metadata_models
from src.dataset import load_split, make_loader
from src.evaluate import load_model
from src.train import predict
from src.utils import (DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT, get_device, load_json, resolve_data_root,
                       save_json, set_seed)

DENSE_RUNS = {"densenet_s42": "densenet121_pad_seed42", "densenet_s43": "densenet121_pad_seed43",
              "densenet_s44": "densenet121_pad_seed44"}
BASE_RUNS = {"pad_s42": "baseline_aspect_20ep", "resize_s42": "baseline_resize_20ep",
             "pad_s43": "baseline_pad_seed43", "resize_s43": "baseline_resize_seed43",
             "pad_s44": "baseline_pad_seed44", "resize_s44": "baseline_resize_seed44"}
AUDIT_DIR6 = DEFAULT_OUTPUT_ROOT / "size_matched_stress_test_6runs"
AUC_TOL = 1e-4
GROUP_COLOR = {"densenet121": "#2a9d6f", "baseline_pad": "#d1603d", "baseline_resize": "#3b6fb6"}


def group_of(series, info):
    m = info[series]["mode"]
    return {"densenet": "densenet121", "pad": "baseline_pad", "resize": "baseline_resize"}.get(m, "metadata")


def plot_group_summary(summ, out_path, title):
    """Mean +- SD (across seeds) of within-subset AUC per group, per subset."""
    subsets = list(dict.fromkeys(summ.label))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    groups = list(GROUP_COLOR)
    w = 0.8 / len(groups)
    for k, g in enumerate(groups):
        t = summ[summ.group == g].set_index("label").reindex(subsets)
        x = np.arange(len(subsets)) - 0.4 + w * (k + 0.5)
        ax.errorbar(x, t["mean"], yerr=t["std"], fmt="o", ms=5, capsize=3, color=GROUP_COLOR[g], label=g)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(np.arange(len(subsets)), subsets, fontsize=6.5)
    ax.set_ylim(0.96, 1.002)
    ax.set_ylabel("ROC-AUC, mean +- sample SD over 3 seeds")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Validation-only size-matched stress test, DenseNet121 seeds 42-44.")
    p.add_argument("--data-root")
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--audit-dir", default=str(DEFAULT_OUTPUT_ROOT / "shortcut_audit"))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "size_matched_stress_test_densenet_all"))
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
    allm.MODE_COLOR["densenet"] = GROUP_COLOR["densenet121"]  # runtime-only; no source file is edited
    print("Validation split only; the test split is not read. Exploratory; inference only.")

    # ---- alignment: manifest <-> metadata
    meta = pd.read_csv(Path(args.audit_dir) / "metadata_val.csv")
    train_meta = pd.read_csv(Path(args.audit_dir) / "metadata_train.csv")
    frame = load_split(args.manifest, root, "val")
    allm.check_alignment(meta, train_meta, frame)
    order = frame.rel_path.tolist()
    meta = meta.set_index("path").loc[order].reset_index()
    print(f"Alignment OK: {len(frame)} val images match the manifest by path, label and patient_id.")

    probs, thresholds, info, align = {}, {}, {}, []
    # ---- DenseNet checkpoints: verify own config, infer with own preprocessing, reproduce saved AUC
    for label, run in DENSE_RUNS.items():
        run_dir = DEFAULT_OUTPUT_ROOT / run
        cfg = load_json(run_dir / "config.json")
        model, ckpt = load_model(run_dir / "best.pt", device)
        pre = ckpt["model"].get("preprocess", "resize")
        want_seed = int(label.split("_s")[-1])
        if (ckpt["model"]["name"] != "densenet121" or cfg["model"] != "densenet121" or pre != "pad"
                or cfg["preprocess"] != pre or ckpt.get("seed") != cfg["seed"] or cfg["seed"] != want_seed
                or cfg.get("pretrained")):
            raise RuntimeError(f"{run}: unexpected configuration (ckpt {ckpt['model']['name']}/{pre}/seed "
                               f"{ckpt.get('seed')}; config {cfg['model']}/{cfg['preprocess']}/seed {cfg['seed']}/"
                               f"pretrained={cfg.get('pretrained')})")
        loader, fr = make_loader(args.manifest, root, "val", args.batch_size, num_workers=0, seed=args.seed,
                                 train=False, pin_memory=device.type == "cuda", preprocess=pre)
        if fr.rel_path.tolist() != order:
            raise RuntimeError(f"{run}: loader order differs from metadata order")
        pr, targets, _ = predict(model, loader, device, amp=device.type == "cuda", desc=label)
        if not (targets == meta.target.values).all():
            raise RuntimeError(f"{run}: label order mismatch")
        pooled, saved = base.auc_or_nan(targets, pr), load_json(run_dir / "val_metrics.json")["roc_auc"]
        if abs(pooled - saved) > AUC_TOL:
            raise RuntimeError(f"{run}: pooled val AUC {pooled:.6f} does not reproduce saved {saved:.6f}")
        probs[label], thresholds[label] = pr, load_json(run_dir / "threshold.json")["threshold"]
        info[label] = {"run": run, "mode": "densenet", "seed": cfg["seed"], "preprocess": pre}
        align.append({"series": label, "run": run, "source": "fresh inference", "pooled_auc_here": pooled,
                      "pooled_auc_saved": saved, "abs_diff": abs(pooled - saved)})
        print(f"{label} ({run}): densenet121/{pre}/seed {cfg['seed']}/pretrained={cfg['pretrained']}; pooled val AUC "
              f"{pooled:.6f} vs saved {saved:.6f}  OK")
        del model
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    # ---- baselines from the six-run audit (identical images by path), pooled AUC re-verified
    pred6 = pd.read_csv(AUDIT_DIR6 / "val_predictions_all_runs.csv").set_index("path")
    if set(pred6.index) != set(order):
        raise RuntimeError("six-run predictions cover a different set of validation images")
    pred6 = pred6.loc[order]
    if not (pred6.label.values == meta.label.values).all() or not (pred6.patient_id.values == meta.patient_id.values).all():
        raise RuntimeError("six-run predictions disagree with manifest labels/patient_ids")
    for label, run in BASE_RUNS.items():
        cfg = load_json(DEFAULT_OUTPUT_ROOT / run / "config.json")
        probs[label] = pred6[f"prob_{label}"].values
        thresholds[label] = load_json(DEFAULT_OUTPUT_ROOT / run / "threshold.json")["threshold"]
        info[label] = {"run": run, "mode": cfg["preprocess"], "seed": cfg["seed"], "preprocess": cfg["preprocess"]}
        pa, sv = base.auc_or_nan(meta.target.values, probs[label]), load_json(DEFAULT_OUTPUT_ROOT / run / "val_metrics.json")["roc_auc"]
        if abs(pa - sv) > AUC_TOL:
            raise RuntimeError(f"{label}: six-run predictions give pooled AUC {pa:.6f}, expected {sv:.6f}")
        align.append({"series": label, "run": run, "source": "six-run audit predictions", "pooled_auc_here": pa,
                      "pooled_auc_saved": sv, "abs_diff": abs(pa - sv)})
    print("Baseline predictions reused from the six-run audit; all six pooled AUCs match their saved metrics.")
    lr = metadata_models(args.seed)["logistic_regression"].fit(train_meta[FEATURES].values, train_meta.target.values)
    probs["metadata_lr"] = lr.predict_proba(meta[FEATURES].values)[:, 1]
    info["metadata_lr"] = {"run": "metadata_lr(fit on train)", "mode": "metadata", "seed": -1, "preprocess": "n/a"}

    # ---- subsets (same construction as the six-run audit), verified against its saved output
    subsets = []
    for by in ("pixels", "height"):
        binned = base.make_bins(meta, by, args.n_bins)
        for b in binned.cat.categories:
            subsets.append(("bin", by, f"{b.left:.0f}-{b.right:.0f}", np.flatnonzero((binned == b).values)))
    matched, n_pairs = base.caliper_match(meta, np.random.RandomState(args.seed), args.caliper)
    subsets.append(("matched", "log_pixels", f"caliper{args.caliper}", meta.index.get_indexer(matched.index)))

    rows = []
    for subset, by, bname, idx in subsets:
        rows += allm.rows_for_subset(meta.iloc[idx], subset, by, bname, {k: v[idx] for k, v in probs.items()},
                                     thresholds, info, args.min_per_class if subset == "bin" else 1,
                                     args.n_boot, args.seed)
    long_df = pd.DataFrame(rows)
    long_df["group"] = [group_of(s, info) for s in long_df.series]

    six = pd.read_csv(AUDIT_DIR6 / "results_long_all_runs.csv")
    for (subset, by, bname), g in long_df.groupby(["subset", "bin_by", "bin"], sort=False):
        s = six[(six.subset == subset) & (six.bin_by == by) & (six.bin == bname)]
        if s.empty or (s.n.iloc[0], s.n_normal.iloc[0], s.n_pneumonia.iloc[0]) != (g.n.iloc[0], g.n_normal.iloc[0], g.n_pneumonia.iloc[0]):
            raise RuntimeError(f"subset {subset}/{by}/{bname} differs from the six-run audit")
        for label in BASE_RUNS:
            a, b6 = g[g.series == label].auc.iloc[0], s[s.series == label].auc.iloc[0]
            if not (np.isnan(a) and np.isnan(b6)) and abs(a - b6) > 1e-9:
                raise RuntimeError(f"{label} AUC in {subset}/{by}/{bname} ({a}) != six-run audit ({b6})")
    print("Bins and matched pairs verified identical to the six-run audit; all six baseline AUCs reproduced exactly.")

    long_df.to_csv(out / "results_long.csv", index=False)
    pd.DataFrame(align).to_csv(out / "alignment_checks.csv", index=False)
    pd.DataFrame({"path": meta.path, "patient_id": meta.patient_id, "label": meta.label, "height": meta.height,
                  "pixels": meta.pixels, **{f"prob_{k}": v for k, v in probs.items()}}
                 ).to_csv(out / "val_predictions_densenet_and_baselines.csv", index=False)

    # ---- per-seed tables
    dense = long_df[long_df.group == "densenet121"]
    for by in ("pixels", "height"):
        t = long_df[(long_df.subset == "bin") & (long_df.bin_by == by)]
        wide = t.drop_duplicates("bin").set_index("bin")[["n", "n_normal", "n_pneumonia", "eligible"]].join(
            t.pivot(index="bin", columns="series", values="auc")).reindex(list(dict.fromkeys(t.bin)))
        wide.to_csv(out / f"bins_by_{by}_auc_wide.csv")
        print(f"\nBins by {by}: ROC-AUC per series (NaN = undefined: fewer than {args.min_per_class} per class)")
        print(wide.round(4).to_string())
        d = dense[(dense.subset == "bin") & (dense.bin_by == by) & dense.eligible][["bin", "series", "auc", "ci_lo", "ci_hi"]]
        print("DenseNet 95% patient-bootstrap CIs:")
        print(d.round(4).to_string(index=False))
    undefined = long_df[~long_df.eligible].drop_duplicates(["bin_by", "bin"])[["bin_by", "bin", "n_normal", "n_pneumonia", "status"]]
    undefined.to_csv(out / "undefined_bins.csv", index=False)
    print("\nBins with undefined AUC (kept in all tables with status):")
    print(undefined.to_string(index=False))

    m = long_df[long_df.subset == "matched"]
    m.to_csv(out / "matched_subset_all_series.csv", index=False)
    print(f"\nMatched subset: {n_pairs} pairs = {2 * n_pairs} images; {len(meta) - 2 * n_pairs} of {len(meta)} images "
          f"excluded (PNEUMONIA excluded {int((meta.target == 1).sum()) - n_pairs}, NORMAL excluded "
          f"{int((meta.target == 0).sum()) - n_pairs}).")
    print(m[["series", "n", "auc", "ci_lo", "ci_hi", "sens_at_saved_thr", "spec_at_saved_thr"]].round(4).to_string(index=False))

    # ---- mean +- sample SD across seeds per group (eligible subsets only)
    ok = long_df[long_df.eligible & (long_df.group != "metadata")].copy()
    ok["label"] = ok.bin_by + ":" + ok.bin
    summ = (ok.groupby(["subset", "bin_by", "bin", "label", "group"], sort=False)
            .agg(mean=("auc", "mean"), std=("auc", "std"), min=("auc", "min"), max=("auc", "max"),
                 seeds=("auc", "count"), n_normal=("n_normal", "first"), n_pneumonia=("n_pneumonia", "first"),
                 sens_mean=("sens_at_saved_thr", "mean"), sens_sd=("sens_at_saved_thr", "std"),
                 spec_mean=("spec_at_saved_thr", "mean"), spec_sd=("spec_at_saved_thr", "std")).reset_index())
    summ.to_csv(out / "group_summary_mean_sd.csv", index=False)
    show = summ.assign(auc=lambda d: d["mean"].round(4).astype(str) + " +- " + d["std"].round(4).astype(str))
    print("\nROC-AUC mean +- sample SD over 3 seeds, by group (eligible subsets):")
    print(show[["label", "n_normal", "n_pneumonia", "group", "auc", "min", "max"]].round(4).to_string(index=False))
    ms = summ[summ.subset == "matched"]
    print("\nMatched subset, sensitivity/specificity at each run's saved threshold, mean +- SD across seeds:")
    print(ms[["group", "sens_mean", "sens_sd", "spec_mean", "spec_sd"]].round(4).to_string(index=False))
    meta_row = long_df[(long_df.series == "metadata_lr") & long_df.eligible][["subset", "bin_by", "bin", "auc"]]
    meta_row.to_csv(out / "metadata_only_auc.csv", index=False)

    # ---- plots
    allm.plot_bins(long_df[long_df.group.isin(["densenet121", "metadata"])], "pixels", out / "auc_by_pixels_bin_densenet_seeds.png")
    allm.plot_bins(long_df[long_df.group.isin(["densenet121", "metadata"])], "height", out / "auc_by_height_bin_densenet_seeds.png")
    allm.plot_matched(long_df[(long_df.subset == "matched") & (long_df.group != "baseline_resize")], out / "matched_subset_auc.png")
    plot_group_summary(summ[summ.subset == "bin"], out / "group_mean_sd_by_bin.png",
                       "Within-bin ROC-AUC, mean +- SD over 3 seeds (validation; exploratory)")
    save_json({"note": "Exploratory; validation only; inference only; test split never read. Size matching does not prove "
                       "absence of shortcuts or clinical generalisation.", "alignment": align, "matched_pairs": n_pairs,
               "included_images": 2 * n_pairs, "excluded_images": len(meta) - 2 * n_pairs,
               "verified_identical_to_six_run_audit": True}, out / "summary.json")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
