import random

import streamlit as st

from dashboard import discovery
from dashboard.charts import bar_chart
from dashboard.discovery import SAFE_SPLITS_FOR_DISPLAY
from dashboard.theme import disclaimer
from src.utils import resolve_data_root


def render():
    st.title("Dataset Explorer")
    disclaimer("The TEST split's images are never opened or displayed by this dashboard; only its "
              "row count (from the manifest, not the image files) is shown below, for context.")

    counts, err = discovery.manifest_split_counts()
    if err:
        st.error(f"Could not read data/splits.csv: {err}")
        return

    st.subheader("Split sizes (from data/splits.csv)")
    totals = counts["totals"]
    st.dataframe(totals.rename("images").to_frame().T, width="stretch")

    st.subheader("Class distribution by split")
    bl = counts["by_split_label"]
    bar_chart(bl.index.tolist(), {c: bl[c].tolist() for c in bl.columns},
             title="NORMAL vs PNEUMONIA by split", y_title="images")

    st.subheader("Browse sample images")
    st.caption("Limited to the train and validation splits by this project's test-split policy.")
    split = st.selectbox("Split", SAFE_SPLITS_FOR_DISPLAY, key="browse_split")
    label = st.selectbox("Label", ["NORMAL", "PNEUMONIA"], key="browse_label")
    n = st.slider("Number of images", 1, 12, 6, key="browse_n")

    data_root_input = st.text_input(
        "Dataset root (folder that CONTAINS chest_xray/)",
        value=st.session_state.get("data_root", ""), key="data_root_input",
        help="Needed to actually display image files; manifest-only info above doesn't need this.")
    if not data_root_input:
        st.info("Enter the dataset root above to browse sample images.")
        return
    try:
        root = resolve_data_root(data_root_input)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        return
    st.session_state["data_root"] = str(root)

    try:
        from src.dataset import load_split
        frame = load_split(discovery.DEFAULT_MANIFEST, root, split)
    except Exception as e:
        st.error(f"Could not load the {split} split from this dataset root: {e}")
        return
    sub = frame[frame.label == label]
    if sub.empty:
        st.info(f"No {label} images in the {split} split.")
        return
    sample = sub.sample(min(n, len(sub)), random_state=random.Random(42).randint(0, 10_000))
    cols = st.columns(min(n, 6))
    for i, row in enumerate(sample.itertuples()):
        with cols[i % len(cols)]:
            st.image(row.abs_path, caption=f"{split}/{label}", width="stretch")
