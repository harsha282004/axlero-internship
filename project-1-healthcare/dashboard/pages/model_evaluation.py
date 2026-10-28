import streamlit as st

from dashboard import discovery
from dashboard.charts import bar_chart, confusion_matrix_heatmap, line_chart
from dashboard.theme import disclaimer, fmt_num, fmt_pct, metric_card
from src.utils import CLASS_NAMES


def render():
    st.title("Model Evaluation")
    disclaimer("All metrics below are read from each run's saved VALIDATION-split "
              "val_metrics.json/threshold.json -- nothing is retrained or re-evaluated here, "
              "and the TEST split is never read.")

    runs = discovery.list_checkpoint_runs()
    if not runs:
        st.info("No trained checkpoints (runs/<name>/best.pt + config.json) were found.")
        return
    run = st.selectbox("Experiment / run", runs, key="eval_run")
    s = discovery.run_summary(run)

    st.caption(f"Checkpoint: `{s['run_dir'] / 'best.pt'}`")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"**Model**\n\n{s['model'] or 'N/A'}")
    with c2:
        st.markdown(f"**Preprocess**\n\n{s['preprocess'] or 'N/A'}")
    with c3:
        st.markdown(f"**Pad aug**\n\n{s['pad_aug']}")
    with c4:
        st.markdown(f"**Edge aug**\n\n{s['edge_aug']}")

    vm = s["val_metrics"]
    if vm is None:
        st.warning("No val_metrics.json for this run -- it may not have finished training.")
        return

    st.subheader("Validation metrics")
    labels = [("Accuracy", "accuracy", fmt_pct), ("ROC-AUC", "roc_auc", fmt_num),
             ("PR-AUC", "pr_auc", fmt_num), ("Sensitivity", "sensitivity", fmt_pct),
             ("Specificity", "specificity", fmt_pct), ("Precision", "precision", fmt_pct),
             ("F1", "f1", fmt_num)]
    cols = st.columns(4)
    for i, (label, key, fmt) in enumerate(labels):
        with cols[i % 4]:
            metric_card(label, fmt(vm.get(key)))

    if vm.get("confusion_matrix"):
        st.subheader("Validation confusion matrix")
        confusion_matrix_heatmap(vm["confusion_matrix"], CLASS_NAMES,
                                 title=f"{run} -- validation (threshold={vm.get('threshold', 'N/A')})")

    thr = s["threshold"]
    if thr:
        st.caption(f"Decision threshold: {thr.get('threshold'):.4f} "
                  f"(strategy={thr.get('strategy', thr.get('selected_on'))}, selected on VALIDATION)")

    hist = s["history"]
    if hist is not None and not hist.empty:
        st.subheader("Training history (validation metrics per epoch)")
        series = {}
        for col in ("val_roc_auc", "val_sensitivity@0.5", "val_specificity@0.5"):
            if col in hist:
                series[col] = hist[col].tolist()
        if series:
            line_chart(hist["epoch"].tolist(), series, title="Validation metrics across training",
                      y_title="score")
        if {"train_loss", "val_loss"}.issubset(hist.columns):
            line_chart(hist["epoch"].tolist(),
                     {"train_loss": hist["train_loss"].tolist(), "val_loss": hist["val_loss"].tolist()},
                     title="Loss across training", y_title="loss")

    fam = discovery.family_validation_table()
    if not fam.empty and "roc_auc_mean" in fam:
        st.subheader("Compare across discovered experiments (validation, mean +/- SD over seeds)")
        show = fam[["model", "preprocess", "pad_aug", "edge_aug",
                   "roc_auc_mean", "roc_auc_std", "sensitivity_mean", "specificity_mean",
                   "roc_auc_count"]].rename(columns={"roc_auc_count": "n_seeds"})
        st.dataframe(show.round(4), width="stretch")
        cats = [f"{r.model}/{r.pad_aug}/{r.edge_aug}" for r in fam.itertuples()]
        bar_chart(cats, {"ROC-AUC": fam.roc_auc_mean.tolist(),
                        "Sensitivity": fam.sensitivity_mean.tolist(),
                        "Specificity": fam.specificity_mean.tolist()},
                 title="Validation metrics across all discovered configurations", y_title="score")
