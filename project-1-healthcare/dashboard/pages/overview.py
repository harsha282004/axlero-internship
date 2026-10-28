import streamlit as st

from dashboard import discovery
from dashboard.charts import bar_chart, pie_chart
from dashboard.theme import disclaimer, fmt_num, fmt_pct, metric_card


def render():
    st.title("Chest X-Ray Pneumonia Classifier")
    st.caption("Project 1: Healthcare Image Classification -- Axlero Data Science & ML Internship")
    disclaimer("Educational / research project. Not a diagnostic tool. Not clinically validated. "
              "All metrics shown are measured on one patient-level split of one public dataset.")

    st.subheader("Dataset")
    counts, err = discovery.manifest_split_counts()
    if err:
        st.warning(f"Could not read the dataset manifest: {err}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total images", f"{counts['n_total']:,}")
        with c2:
            metric_card("Patients", f"{counts['n_patients']:,}")
        with c3:
            metric_card("Train images", f"{int(counts['totals'].get('train', 0)):,}")
        with c4:
            metric_card("Validation images", f"{int(counts['totals'].get('val', 0)):,}")
        st.caption("Test-split size is shown on the Dataset Explorer page; test images/predictions "
                  "are never opened by this dashboard.")
        bl = counts["by_split_label"]
        cols = st.columns(2)
        with cols[0]:
            pie_chart(bl.columns.tolist(), bl.loc["train"].tolist() if "train" in bl.index else [],
                     title="Train split: class balance")
        with cols[1]:
            pie_chart(bl.columns.tolist(), bl.loc["val"].tolist() if "val" in bl.index else [],
                     title="Validation split: class balance")

    st.subheader("Model & preprocessing")
    from src.dataset import IMAGE_SIZE, MEAN, STD
    st.markdown(f"""
- **Architectures:** `baseline` (small from-scratch CNN) and `densenet121`
  (torchvision DenseNet121, trained from scratch, one logit = P(PNEUMONIA)).
- **Preprocessing:** `Grayscale(3)` &rarr; aspect-preserving resize-and-pad to
  **{IMAGE_SIZE}x{IMAGE_SIZE}** (or squash-resize, depending on the run) &rarr; `ToTensor`
  &rarr; fixed ImageNet normalisation (mean {MEAN}, std {STD}).
- **Training-only augmentations** (both default-off, independent, composable):
  `RandomPadLetterbox` (randomised letterbox fill/offset) and `RandomEdgeMask`
  (randomised outer-border crop/mask, 0-10% of the shorter side).
""")

    st.subheader("Available experiments")
    runs = discovery.list_checkpoint_runs()
    audits = discovery.list_audit_dirs()
    c1, c2 = st.columns(2)
    with c1:
        metric_card("Trained checkpoints discovered", len(runs))
    with c2:
        metric_card("Robustness/analysis outputs discovered", len(audits))

    st.subheader("Selected experiment -- validation metrics")
    if not runs:
        st.info("No trained checkpoints found under runs/.")
        return
    default_idx = next((i for i, r in enumerate(runs) if "edgeaug" in r), 0)
    run = st.selectbox("Choose an experiment", runs, index=default_idx, key="overview_run")
    s = discovery.run_summary(run)
    vm = s["val_metrics"]
    st.caption(f"Checkpoint: `{s['run_dir'] / 'best.pt'}` -- model={s['model']}, "
              f"preprocess={s['preprocess']}, pad_aug={s['pad_aug']}, edge_aug={s['edge_aug']}, "
              f"seed={s['seed']}")
    if vm is None:
        st.warning("This run has no val_metrics.json yet.")
        return
    cols = st.columns(4)
    for col, (label, key, fmt) in zip(
        cols * 2,
        [("ROC-AUC", "roc_auc", fmt_num), ("PR-AUC", "pr_auc", fmt_num),
         ("Sensitivity", "sensitivity", fmt_pct), ("Specificity", "specificity", fmt_pct),
         ("Precision", "precision", fmt_pct), ("F1", "f1", fmt_num),
         ("Accuracy", "accuracy", fmt_pct)]):
        with col:
            metric_card(label, fmt(vm.get(key)))

    fam = discovery.family_validation_table()
    if not fam.empty and "roc_auc_mean" in fam:
        st.subheader("DenseNet121 family comparison (validation, mean across seeds)")
        dn = fam[fam.model == "densenet121"]
        if not dn.empty:
            cats = [f"pad={r.pad_aug}/edge={r.edge_aug}" for r in dn.itertuples()]
            bar_chart(cats, {"ROC-AUC": dn.roc_auc_mean.tolist(),
                             "Sensitivity": dn.sensitivity_mean.tolist(),
                             "Specificity": dn.specificity_mean.tolist()},
                     title="Validation metrics by augmentation configuration", y_title="score")
