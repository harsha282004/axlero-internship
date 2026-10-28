import streamlit as st

from dashboard import discovery
from dashboard.charts import bar_chart, line_chart
from dashboard.theme import disclaimer


def render():
    st.title("Robustness Analysis")
    disclaimer(
        "All results on this page are VALIDATION-split-only, exploratory, SYNTHETIC perturbation "
        "experiments (hand-designed masking/cropping/padding-fill changes). A change under a "
        "synthetic perturbation is NOT proof of shortcut use and does NOT establish real-world or "
        "clinical robustness -- masking can create out-of-distribution inputs or remove genuine "
        "anatomy. These are pre-computed results read from disk; this page never (re)runs an "
        "audit script.")

    st.subheader("Family comparison: original vs. padding-augmented vs. EdgeAug")
    merged = discovery.margin_family_comparison()
    if merged is None:
        st.info("The three-family margin-sensitivity comparison files were not found or had an "
               "unexpected format -- see the per-audit results below instead.")
    else:
        metric = st.selectbox("Metric", sorted(merged["metric"].unique()),
                             index=max(0, sorted(merged["metric"].unique()).index("specificity"))
                             if "specificity" in merged.metric.values else 0,
                             key="margin_metric")
        sub = merged[merged.metric == metric]
        bar_chart(sub.condition.tolist(),
                 {"orig": sub.orig_mean_delta.tolist(), "aug": sub.aug_mean_delta.tolist(),
                  "edgeaug": sub.edgeaug_mean_delta.tolist()},
                 title=f"Mean change in {metric} vs. each checkpoint's own control, by condition",
                 y_title=f"delta {metric}")
        st.caption("Each bar is the mean, over 3 seeds, of (perturbed - control) at that "
                  "checkpoint's own fixed, saved validation threshold -- never retuned per "
                  "condition. Source: runs/margin_sensitivity_audit(_edgeaug)/family_comparison*.csv")

    st.subheader("Size-matched stress test: matched-subset ROC-AUC")
    sm = discovery.size_matched_family_comparison()
    if sm is None:
        st.info("Size-matched comparison files were not found for enough families to compare.")
    else:
        fams = list(sm)
        bar_chart(["matched subset"], {f: [sm[f]["mean"]] for f in fams},
                 title="ROC-AUC on identical caliper-matched, size-balanced image pairs",
                 y_title="ROC-AUC")
        st.caption("Mean +/- sample SD over 3 seeds, on the exact same matched pairs across "
                  "families: " + ", ".join(f"{f}: {sm[f]['mean']:.4f} +/- {sm[f]['std']:.4f}"
                                           for f in fams))

    st.subheader("All discovered robustness / analysis outputs")
    audits = discovery.list_audit_dirs()
    if not audits:
        st.info("No audit output directories found under runs/.")
        return
    for name in audits:
        a = discovery.audit_summary(name)
        label = discovery.KNOWN_AUDIT_LABELS.get(name, name)
        with st.expander(label):
            st.caption(f"`runs/{name}/`")
            summ = a["summary"] or {}
            if "note" in summ:
                st.markdown(f"*{summ['note']}*")
            scalar_keys = [k for k, v in summ.items()
                          if k != "note" and isinstance(v, (int, float, str, bool))]
            if scalar_keys:
                st.json({k: summ[k] for k in scalar_keys})
            if a["pngs"]:
                cols = st.columns(min(3, len(a["pngs"])))
                for i, p in enumerate(a["pngs"][:3]):
                    with cols[i % len(cols)]:
                        st.image(str(p), caption=p.name, width="stretch")
                if len(a["pngs"]) > 3:
                    st.caption(f"+{len(a['pngs']) - 3} more figure(s) in this folder.")
