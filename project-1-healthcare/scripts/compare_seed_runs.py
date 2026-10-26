"""Compare saved baseline runs across seeds and preprocessing modes (read-only on run artifacts).

Reads config.json, history.csv, val_metrics.json and threshold.json of each run. Validation only;
no test split, no inference, no training. Writes tables to runs/seed_comparison/ (git-ignored).

    .venv\\Scripts\\python.exe scripts\\compare_seed_runs.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils import DEFAULT_OUTPUT_ROOT, load_json, save_json

RUNS = ["baseline_aspect_20ep", "baseline_pad_seed43", "baseline_pad_seed44",
        "baseline_resize_20ep", "baseline_resize_seed43", "baseline_resize_seed44"]
METRICS = ["best_epoch", "threshold", "roc_auc", "pr_auc", "val_loss", "sensitivity", "specificity",
           "precision", "f1", "accuracy"]
SAME_CONFIG_KEYS = ["model", "epochs", "batch_size", "lr", "weight_decay", "dropout", "patience", "amp",
                    "select_metric", "threshold_strategy", "pos_weight", "no_class_weights", "num_workers",
                    "no_deterministic"]


def load_run(name):
    d = DEFAULT_OUTPUT_ROOT / name
    cfg, thr, met = load_json(d / "config.json"), load_json(d / "threshold.json"), load_json(d / "val_metrics.json")
    hist = pd.read_csv(d / "history.csv")
    best = hist[hist.epoch == thr["best_epoch"]].iloc[0]
    row = {"run": name, "preprocess": cfg["preprocess"], "seed": cfg["seed"], "epochs_run": len(hist),
           "best_epoch": thr["best_epoch"], "threshold": thr["threshold"], "val_loss": float(best.val_loss)}
    row.update({k: met[k] for k in ("roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "accuracy")})
    row["confusion_matrix"] = str(met["confusion_matrix"])
    return row, cfg


def main():
    rows, cfgs = [], {}
    for r in RUNS:
        row, cfg = load_run(r)
        rows.append(row)
        cfgs[r] = cfg
    df = pd.DataFrame(rows)

    # configuration consistency across all runs (only preprocess and seed should differ)
    print("Configuration consistency (keys that must be identical across runs):")
    for k in SAME_CONFIG_KEYS:
        vals = {str(c.get(k)) for c in cfgs.values()}
        print(f"  {k:20s} {'OK  ' if len(vals) == 1 else 'DIFF'} {sorted(vals)}")
    print("  preprocess/seed per run:", {r: (c["preprocess"], c["seed"]) for r, c in cfgs.items()})

    out = DEFAULT_OUTPUT_ROOT / "seed_comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_run.csv", index=False)
    show = df.drop(columns=["confusion_matrix"]).round(4)
    print("\nPer-run results (best validation epoch; threshold selected on validation):")
    print(show.to_string(index=False))

    agg = df.groupby("preprocess")[METRICS].agg(["mean", "std"]).round(4)
    agg.to_csv(out / "mean_std_by_preprocess.csv")
    print("\nMean and sample SD (ddof=1) across 3 seeds [42, 43, 44] per preprocessing mode:")
    for mode, g in df.groupby("preprocess"):
        print(f"  {mode} (n={len(g)}): " + "; ".join(
            f"{m} {g[m].mean():.4f}±{g[m].std():.4f}" for m in METRICS))
    diff = (df[df.preprocess == "pad"][METRICS].mean() - df[df.preprocess == "resize"][METRICS].mean()).round(4)
    print("\nDifference of means (pad - resize):", diff.to_dict())
    save_json({"per_run": rows, "note": "Three seeds = preliminary variability estimate; no significance claim."},
              out / "summary.json")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
