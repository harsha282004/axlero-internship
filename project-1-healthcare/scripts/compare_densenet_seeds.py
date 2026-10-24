"""Compare DenseNet121 seeds 42/43/44 with the three-seed baseline CNNs (saved artifacts only).

Reads config.json, history.csv, val_metrics.json and threshold.json. Validation only; no inference,
no test split, no training. Reuses load_run from scripts/compare_seed_runs.py (unchanged).
Writes tables to runs/densenet_seed_comparison/ (git-ignored).

    .venv\\Scripts\\python.exe scripts\\compare_densenet_seeds.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd

from compare_seed_runs import METRICS, SAME_CONFIG_KEYS, load_run  # existing, unchanged
from src.utils import DEFAULT_OUTPUT_ROOT, save_json

DENSE = ["densenet121_pad_seed42", "densenet121_pad_seed43", "densenet121_pad_seed44"]
BASE_PAD = ["baseline_aspect_20ep", "baseline_pad_seed43", "baseline_pad_seed44"]
BASE_RESIZE = ["baseline_resize_20ep", "baseline_resize_seed43", "baseline_resize_seed44"]
# lr and batch size are the only intended differences between DenseNet and baseline configs
DENSE_FIXED = ["model", "epochs", "batch_size", "lr", "weight_decay", "dropout", "patience", "amp", "pretrained",
               "preprocess", "select_metric", "threshold_strategy", "pos_weight", "no_class_weights", "num_workers",
               "no_deterministic", "freeze_backbone"]


def main():
    groups = {"densenet121_pad": DENSE, "baseline_pad": BASE_PAD, "baseline_resize": BASE_RESIZE}
    rows, cfgs = [], {}
    for g, runs in groups.items():
        for r in runs:
            row, cfg = load_run(r)
            row["group"] = g
            rows.append(row)
            cfgs[r] = cfg
    df = pd.DataFrame(rows)

    print("DenseNet configuration consistency (must be identical across the three DenseNet runs except seed):")
    for k in DENSE_FIXED:
        vals = {str(cfgs[r].get(k)) for r in DENSE}
        print(f"  {k:20s} {'OK  ' if len(vals) == 1 else 'DIFF'} {sorted(vals)}")
    print("  seeds:", [cfgs[r]["seed"] for r in DENSE])
    print("DenseNet vs baseline differences (intended: model, lr, batch size):")
    for k in ("model", "lr", "batch_size", "weight_decay", "dropout", "epochs", "patience", "amp", "pos_weight"):
        print(f"  {k:14s} densenet={cfgs[DENSE[0]].get(k)}  baseline={cfgs[BASE_PAD[0]].get(k)}")

    out = DEFAULT_OUTPUT_ROOT / "densenet_seed_comparison"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_run.csv", index=False)
    print("\nPer-run results (best validation epoch; threshold selected on validation):")
    print(df[df.group == "densenet121_pad"].drop(columns=["confusion_matrix", "group"]).round(4).to_string(index=False))

    agg = df.groupby("group")[METRICS].agg(["mean", "std"])
    agg.round(4).to_csv(out / "mean_std_by_group.csv")
    print("\nMean and sample SD (ddof=1) across seeds 42/43/44:")
    for g in groups:
        s = df[df.group == g]
        print(f"  {g} (n={len(s)}): " + "; ".join(f"{m} {s[m].mean():.4f}+-{s[m].std():.4f}" for m in METRICS))
    print("\nRange (min..max) across seeds:")
    for g in groups:
        s = df[df.group == g]
        print(f"  {g}: " + "; ".join(f"{m} {s[m].min():.4f}..{s[m].max():.4f}" for m in
                                     ("roc_auc", "pr_auc", "val_loss", "sensitivity", "specificity", "f1")))
    d = (df[df.group == "densenet121_pad"][METRICS].mean() - df[df.group == "baseline_pad"][METRICS].mean()).round(4)
    print("\nDifference of means, DenseNet - baseline pad:", d.to_dict())
    print("Confusion matrices (rows true NORMAL/PNEUMONIA):",
          {r["run"]: r["confusion_matrix"] for r in rows if r["group"] == "densenet121_pad"})
    save_json({"per_run": rows, "note": "Three seeds each; fixed validation split; preliminary variability only."},
              out / "summary.json")
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
