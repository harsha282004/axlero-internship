"""Read-only results dashboard for Project 1: Healthcare Image Classification.

Launch (from project-1-healthcare/, with the venv activated):
    .venv\\Scripts\\streamlit.exe run dashboard.py

This app only READS existing files under runs/ and data/splits.csv, and runs single-image
inference on user uploads for the X-Ray Prediction page. It never trains or retrains a
model, never writes into runs/, never re-runs an audit/stress-test script, and never
opens the TEST split's images, predictions, or metrics.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from dashboard.pages import (dataset_explorer, experiment_explorer, model_evaluation,  # noqa: E402
                             overview, robustness, xray_prediction)
from dashboard.theme import inject_css  # noqa: E402

PAGES = {
    "Overview": overview,
    "Dataset Explorer": dataset_explorer,
    "Model Evaluation": model_evaluation,
    "X-Ray Prediction": xray_prediction,
    "Robustness Analysis": robustness,
    "Experiment Explorer": experiment_explorer,
}


def main():
    # page_icon uses a plain, long-standing emoji codepoint (not a newer multi-codepoint
    # one) -- some Streamlit/browser combinations have failed to resolve very new emoji
    # (e.g. the X-ray emoji, Unicode 14/2021) into a favicon, which can abort the script
    # run before anything else renders.
    try:
        st.set_page_config(page_title="CXR Classifier Dashboard", page_icon="\U0001F3E5",
                           layout="wide", initial_sidebar_state="expanded")
    except Exception:
        pass  # e.g. called more than once during a hot-reload; page still renders fine
    try:
        inject_css()
    except Exception as e:
        st.warning(f"Could not apply dashboard styling (continuing without it): {e}")

    choice = "Overview"
    try:
        with st.sidebar:
            st.markdown("## Chest X-Ray Classifier")
            st.caption("Educational project -- not clinically validated.")
            choice = st.radio("Navigate", list(PAGES), key="nav")
            st.divider()
            st.caption("Project: `project-1-healthcare/`")
            st.caption("All data read-only from `runs/` and `data/splits.csv`. "
                      "TEST split is never opened.")
    except Exception as e:
        st.error(f"Sidebar navigation failed to render: {e}")

    try:
        PAGES[choice].render()
    except Exception as e:
        st.error(f"This page hit an error and could not finish rendering: {e}")
        with st.expander("Details"):
            st.exception(e)


if __name__ == "__main__":
    main()
