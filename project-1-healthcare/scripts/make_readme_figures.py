"""Generate the small set of summary figures used in README.md, from EXISTING saved
VALIDATION artifacts only (val_metrics.json, and the margin-sensitivity / size-matched
audit CSVs). Read-only: no training, no inference, no test-split file is opened, and no
run or audit output directory is written to -- figures go to docs/images/ only.

Run from project-1-healthcare/ (no --data-root needed; everything here is already on disk):
    .venv\\Scripts\\python.exe scripts\\make_readme_figures.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import DEFAULT_OUTPUT_ROOT, load_json

OUT_DIR = PROJECT_ROOT / "docs" / "images"
DENSENET_FAMILIES = {"orig": "densenet121_pad_seed{seed}", "aug": "densenet121_padaug_seed{seed}",
                     "edgeaug": "densenet121_edgeaug_seed{seed}"}
SEEDS = (42, 43, 44)
COLORS = {"orig": "#2563eb", "aug": "#d97706", "edgeaug": "#16a34a"}


def family_validation_metrics():
    """Mean +/- sample SD, over seeds 42/43/44, of each DenseNet121 family's own saved
    VALIDATION val_metrics.json. Returns a DataFrame indexed by family."""
    rows = []
    for fam, tmpl in DENSENET_FAMILIES.items():
        for seed in SEEDS:
            run_dir = DEFAULT_OUTPUT_ROOT / tmpl.format(seed=seed)
            vm = load_json(run_dir / "val_metrics.json")
            rows.append({"family": fam, "seed": seed, **vm})
    df = pd.DataFrame(rows)
    agg = df.groupby("family")[["roc_auc", "sensitivity", "specificity"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reindex(list(DENSENET_FAMILIES))


def plot_family_validation_metrics(df, path):
    metrics = [("roc_auc", "ROC-AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity")]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(metrics))
    width = 0.8 / len(df)
    for i, fam in enumerate(df.index):
        means = [df.loc[fam, f"{m}_mean"] for m, _ in metrics]
        stds = [df.loc[fam, f"{m}_std"] for m, _ in metrics]
        ax.bar(x + (i - (len(df) - 1) / 2) * width, means, width, yerr=stds, capsize=3,
              label=fam, color=COLORS.get(fam))
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0.95, 1.01)
    ax.set_ylabel("score (validation split)")
    ax.set_title("DenseNet121 validation metrics by augmentation family\n"
                "(mean +/- sample SD over seeds 42/43/44)")
    ax.legend(title="family")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def robustness_specificity_deltas():
    """Mean specificity delta vs. each checkpoint's own control, for two margin-masking
    conditions, from the existing margin-sensitivity audit outputs (orig/aug from
    margin_sensitivity_audit/family_comparison.csv, edgeaug from the edgeaug audit's
    family_comparison_edgeaug_vs_orig.csv). Returns None if the files are unavailable."""
    aug_path = DEFAULT_OUTPUT_ROOT / "margin_sensitivity_audit" / "family_comparison.csv"
    edge_path = (DEFAULT_OUTPUT_ROOT / "margin_sensitivity_audit_edgeaug" /
                "family_comparison_edgeaug_vs_orig.csv")
    if not (aug_path.is_file() and edge_path.is_file()):
        return None
    aug_df = pd.read_csv(aug_path)
    edge_df = pd.read_csv(edge_path)
    conditions = ["margin10_black", "margin10_imgmean"]
    out = {}
    for cond in conditions:
        a = aug_df[(aug_df.condition == cond) & (aug_df.metric == "specificity")]
        e = edge_df[(edge_df.condition == cond) & (edge_df.metric == "specificity")]
        if a.empty or e.empty:
            return None
        out[cond] = {"orig": float(a.orig_mean_delta.iloc[0]), "aug": float(a.aug_mean_delta.iloc[0]),
                     "edgeaug": float(e.edgeaug_mean_delta.iloc[0])}
    return out


def plot_robustness_comparison(deltas, path):
    conditions = list(deltas)
    families = ["orig", "aug", "edgeaug"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(conditions))
    width = 0.8 / len(families)
    for i, fam in enumerate(families):
        vals = [deltas[c][fam] for c in conditions]
        ax.bar(x + (i - (len(families) - 1) / 2) * width, vals, width, label=fam, color=COLORS[fam])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, conditions)
    ax.set_ylabel("mean change in specificity vs. own control")
    ax.set_title("Specificity under synthetic margin-masking, by family\n"
                "(validation split, fixed saved thresholds; exploratory)")
    ax.legend(title="family")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Validation-split artifacts only; no training, inference, or test-split access.")

    fam_df = family_validation_metrics()
    plot_family_validation_metrics(fam_df, OUT_DIR / "family-validation-metrics.png")

    deltas = robustness_specificity_deltas()
    if deltas is None:
        print("Skipping robustness-comparison.png: the required margin-sensitivity audit "
             "CSVs were not found.")
    else:
        plot_robustness_comparison(deltas, OUT_DIR / "robustness-comparison.png")


if __name__ == "__main__":
    main()
