"""Read-only discovery of training runs and audit outputs under runs/.

Every function here only reads files that already exist (JSON/CSV/PNG/best.pt); nothing
is written, retrained, or re-evaluated. Nothing here opens, counts, or reads the TEST
split's images or predictions -- only VALIDATION and manifest-level (row count) data.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils import DEFAULT_MANIFEST, DEFAULT_OUTPUT_ROOT

RUNS_DIR = DEFAULT_OUTPUT_ROOT
SAFE_SPLITS_FOR_DISPLAY = ("train", "val")  # the TEST split is never surfaced for browsing


@st.cache_data(show_spinner=False)
def load_json_safe(path):
    """Read a JSON file; return None if missing or unreadable (never raises into the UI)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
        return None


@st.cache_data(show_spinner=False)
def list_training_runs():
    """Training runs = subfolders of runs/ with a config.json (written only by src.train).

    Audit/stress-test/gradcam output folders never contain config.json, so this cleanly
    separates "a trained checkpoint" from "an analysis output" without hard-coding names.
    """
    if not RUNS_DIR.is_dir():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir()):
        if d.is_dir() and (d / "config.json").is_file():
            out.append(d.name)
    return out


@st.cache_data(show_spinner=False)
def run_summary(run_name):
    """Everything about one training run that can be read without touching test data."""
    run_dir = RUNS_DIR / run_name
    cfg = load_json_safe(run_dir / "config.json") or {}
    val_metrics = load_json_safe(run_dir / "val_metrics.json")
    threshold = load_json_safe(run_dir / "threshold.json")
    history = load_csv_safe(run_dir / "history.csv")
    return {
        "run_name": run_name,
        "run_dir": run_dir,
        "has_checkpoint": (run_dir / "best.pt").is_file(),
        "config": cfg,
        "val_metrics": val_metrics,
        "threshold": threshold,
        "history": history,
        "model": cfg.get("model"),
        "preprocess": cfg.get("preprocess"),
        "pad_aug": cfg.get("pad_aug", "none"),
        "edge_aug": cfg.get("edge_aug", "none"),
        "seed": cfg.get("seed"),
    }


@st.cache_data(show_spinner=False)
def list_checkpoint_runs():
    """Training runs that actually have a usable best.pt (for the evaluation/predict pickers)."""
    return [r for r in list_training_runs() if (RUNS_DIR / r / "best.pt").is_file()]


# Known audit/stress-test output directories this dashboard knows how to label nicely.
# Anything else with a summary.json is still discovered and shown generically.
KNOWN_AUDIT_LABELS = {
    "margin_sensitivity_audit": "Margin-sensitivity audit (orig vs. padding-augmented)",
    "margin_sensitivity_audit_edgeaug": "Margin-sensitivity audit (orig vs. padding-aug vs. EdgeAug)",
    "perturbation_audit_densenet": "Perturbation audit -- orig DenseNet121",
    "perturbation_audit_padaug": "Perturbation audit -- padding-augmented DenseNet121",
    "perturbation_audit_edgeaug": "Perturbation audit -- EdgeAug DenseNet121",
    "size_matched_stress_test_densenet_all": "Size-matched stress test -- orig DenseNet121",
    "size_matched_stress_test_padaug": "Size-matched stress test -- padding-augmented DenseNet121",
    "size_matched_stress_test_edgeaug": "Size-matched stress test -- EdgeAug DenseNet121",
    "shortcut_audit": "Dataset shortcut-learning audit (metadata-only classifiers)",
}


@st.cache_data(show_spinner=False)
def list_audit_dirs():
    """Any runs/ subfolder with a summary.json and no config.json (i.e. not a training run)."""
    if not RUNS_DIR.is_dir():
        return []
    out = []
    for d in sorted(RUNS_DIR.iterdir()):
        if d.is_dir() and (d / "summary.json").is_file() and not (d / "config.json").is_file():
            out.append(d.name)
    return out


METRIC_KEYS = ("roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "accuracy")


@st.cache_data(show_spinner=False)
def family_validation_table():
    """Group checkpoint runs by (model, preprocess, pad_aug, edge_aug) and average their
    VALIDATION metrics across seeds. Discovered dynamically from config.json -- no run-name
    template is assumed, so a new family with a different naming convention is still grouped
    correctly as long as src.train wrote its usual config.json/val_metrics.json."""
    rows = []
    for run in list_checkpoint_runs():
        s = run_summary(run)
        vm = s["val_metrics"]
        if vm is None:
            continue
        rows.append({"run": run, "model": s["model"], "preprocess": s["preprocess"],
                     "pad_aug": s["pad_aug"], "edge_aug": s["edge_aug"], "seed": s["seed"],
                     **{k: vm.get(k) for k in METRIC_KEYS}})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = ["model", "preprocess", "pad_aug", "edge_aug"]
    agg = df.groupby(group_cols, dropna=False)[list(METRIC_KEYS)].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reset_index()


@st.cache_data(show_spinner=False)
def audit_summary(audit_name):
    d = RUNS_DIR / audit_name
    return {"name": audit_name, "dir": d, "summary": load_json_safe(d / "summary.json"),
            "pngs": sorted(d.glob("*.png"))}


@st.cache_data(show_spinner=False)
def margin_family_comparison():
    """Best-effort orig vs. aug vs. edgeaug comparison from the margin-sensitivity audits.

    Returns None if either source file is missing or doesn't have the expected columns --
    the robustness page falls back to the generic per-audit view in that case."""
    try:
        aug_df = load_csv_safe(RUNS_DIR / "margin_sensitivity_audit" / "family_comparison.csv")
        edge_df = load_csv_safe(RUNS_DIR / "margin_sensitivity_audit_edgeaug" /
                                "family_comparison_edgeaug_vs_orig.csv")
        if aug_df is None or edge_df is None:
            return None
        need_aug = {"condition", "metric", "orig_mean_delta", "aug_mean_delta"}
        need_edge = {"condition", "metric", "orig_mean_delta", "edgeaug_mean_delta"}
        if not need_aug.issubset(aug_df.columns) or not need_edge.issubset(edge_df.columns):
            return None
        merged = aug_df[["condition", "metric", "orig_mean_delta", "aug_mean_delta"]].merge(
            edge_df[["condition", "metric", "edgeaug_mean_delta"]], on=["condition", "metric"],
            how="inner")
        return merged if not merged.empty else None
    except Exception:
        return None


SIZE_MATCHED_FAMILY_DIRS = {"orig": "size_matched_stress_test_densenet_all",
                            "aug": "size_matched_stress_test_padaug",
                            "edgeaug": "size_matched_stress_test_edgeaug"}


@st.cache_data(show_spinner=False)
def size_matched_family_comparison():
    """Best-effort matched-subset ROC-AUC (mean +/- SD over seeds) for orig/aug/edgeaug,
    read from each family's own size_matched_stress_test* output. Any family whose output
    is missing or has an unexpected shape is simply omitted, not treated as an error."""
    out = {}
    for fam, dirname in SIZE_MATCHED_FAMILY_DIRS.items():
        df = load_csv_safe(RUNS_DIR / dirname / "group_summary_mean_sd.csv")
        if df is None or not {"subset", "group", "mean", "std"}.issubset(df.columns):
            continue
        row = df[(df.subset == "matched") & (df.group == "densenet121")]
        if row.empty:
            continue
        out[fam] = {"mean": float(row["mean"].iloc[0]), "std": float(row["std"].iloc[0])}
    return out or None


@st.cache_data(show_spinner=False)
def manifest_split_counts():
    """Row counts per split and per label, read from data/splits.csv only (no images opened)."""
    try:
        from src.dataset import read_manifest
        df = read_manifest(DEFAULT_MANIFEST)
    except Exception as e:  # missing/invalid manifest should not crash the whole dashboard
        return None, str(e)
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    totals = df.groupby("split").size()
    return {"by_split_label": counts, "totals": totals, "n_total": len(df),
            "n_patients": df.patient_id.nunique()}, None
