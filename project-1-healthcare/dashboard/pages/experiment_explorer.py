import streamlit as st

from dashboard import discovery
from dashboard.theme import fmt_num, fmt_pct


def render():
    st.title("Experiment Explorer")
    st.caption("Read-only listing of everything discovered under runs/. Nothing here modifies "
              "or deletes any experiment file.")

    runs = discovery.list_training_runs()
    if not runs:
        st.info("No training runs discovered under runs/.")
    else:
        rows = []
        for r in runs:
            s = discovery.run_summary(r)
            vm = s["val_metrics"] or {}
            rows.append({
                "run": r, "checkpoint": "yes" if s["has_checkpoint"] else "no",
                "model": s["model"], "preprocess": s["preprocess"], "pad_aug": s["pad_aug"],
                "edge_aug": s["edge_aug"], "seed": s["seed"],
                "val_roc_auc": vm.get("roc_auc"), "val_sensitivity": vm.get("sensitivity"),
                "val_specificity": vm.get("specificity"),
                "has_history": s["history"] is not None,
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        st.subheader(f"Training runs ({len(df)})")
        st.dataframe(df, width="stretch")

        st.subheader("Compare validation metrics across selected runs")
        chosen = st.multiselect("Runs to compare", runs, default=runs[:min(6, len(runs))],
                                key="compare_runs")
        if chosen:
            comp = df[df.run.isin(chosen)].set_index("run")[
                ["val_roc_auc", "val_sensitivity", "val_specificity"]]
            st.dataframe(comp.round(4).rename(
                columns={"val_roc_auc": "ROC-AUC", "val_sensitivity": "Sensitivity",
                        "val_specificity": "Specificity"}), width="stretch")

        st.subheader("Run details")
        pick = st.selectbox("Inspect one run", runs, key="explorer_run")
        s = discovery.run_summary(pick)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Checkpoint**")
            st.write(f"`{s['run_dir'] / 'best.pt'}`" if s["has_checkpoint"] else "Not found")
            st.markdown("**Configuration (config.json)**")
            st.json(s["config"] or {}, expanded=False)
        with c2:
            st.markdown("**Validation metrics (val_metrics.json)**")
            st.json(s["val_metrics"] or {"status": "not found"}, expanded=False)
            st.markdown("**Decision threshold (threshold.json)**")
            st.json(s["threshold"] or {"status": "not found"}, expanded=False)
        if s["history"] is not None:
            st.markdown("**Training history (history.csv)**")
            st.dataframe(s["history"], width="stretch", height=220)
        st.caption(f"All files for this run live under `{s['run_dir']}` "
                  "(open that folder directly for train.log, history.json, and any eval_val/ "
                  "output; eval_test/, if present, is intentionally not linked from here).")

    st.subheader("Audit / analysis outputs")
    audits = discovery.list_audit_dirs()
    if audits:
        for a in audits:
            st.markdown(f"- `runs/{a}/` -- {discovery.KNOWN_AUDIT_LABELS.get(a, 'analysis output')} "
                       "(see the Robustness Analysis page for a rendered view)")
    else:
        st.info("No audit output directories discovered.")
