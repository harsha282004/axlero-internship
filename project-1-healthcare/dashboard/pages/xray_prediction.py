import streamlit as st

from dashboard import discovery
from dashboard.theme import disclaimer, fmt_pct, metric_card


def render():
    # torch/torchvision/the model loader are imported here, not at module top, so that
    # every OTHER page starts instantly without paying torch's import cost; this page pays
    # it only when actually visited.
    from dashboard.inference import (IncompatibleCheckpointError, InvalidImageError,
                                     load_checkpoint_for_inference, open_uploaded_image, predict_image)

    st.title("X-Ray Prediction")
    disclaimer("This is an educational demo, NOT a medical device. A prediction here is NOT a "
              "diagnosis and must never be used to guide real clinical decisions. Uploaded images "
              "are processed in memory only for this session and are never written to disk.")

    runs = discovery.list_checkpoint_runs()
    if not runs:
        st.info("No trained checkpoints found under runs/.")
        return
    run = st.selectbox("Checkpoint to use", runs, key="predict_run")
    s = discovery.run_summary(run)
    st.caption(f"model={s['model']}, preprocess={s['preprocess']}, pad_aug={s['pad_aug']}, "
              f"edge_aug={s['edge_aug']}, seed={s['seed']}")

    uploaded = st.file_uploader("Upload a chest X-ray (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"],
                               key="predict_upload")
    if uploaded is None:
        st.info("Upload an image to run a prediction.")
        return

    try:
        image = open_uploaded_image(uploaded)
    except InvalidImageError as e:
        st.error(str(e))
        return

    col_img, col_result = st.columns([1, 1])
    with col_img:
        st.image(image, caption=uploaded.name, width="stretch")

    with col_result:
        with st.spinner("Loading checkpoint and running inference..."):
            try:
                model, ckpt, device = load_checkpoint_for_inference(s["run_dir"] / "best.pt")
                result = predict_image(model, ckpt, device, image)
            except IncompatibleCheckpointError as e:
                st.error(f"This checkpoint is not compatible with the dashboard's inference "
                        f"path: {e}")
                return
            except Exception as e:  # malformed image after decode, unexpected tensor shape, etc.
                st.error(f"Could not run a prediction on this image: {e}")
                return

        metric_card("Predicted class", result["predicted_class"])
        c1, c2 = st.columns(2)
        with c1:
            metric_card("P(PNEUMONIA)", fmt_pct(result["prob_pneumonia"]))
        with c2:
            metric_card("P(NORMAL)", fmt_pct(result["prob_normal"]))
        st.caption(result["note"])

        thr = s["threshold"]
        if thr and thr.get("threshold") is not None:
            saved_thr = thr["threshold"]
            saved_pred = "PNEUMONIA" if result["prob_pneumonia"] >= saved_thr else "NORMAL"
            st.caption(f"This run's saved VALIDATION-selected decision threshold is "
                      f"{saved_thr:.4f} ({thr.get('strategy', thr.get('selected_on'))}); at that "
                      f"threshold this image would be classified **{saved_pred}**.")
        else:
            st.caption("No saved decision threshold found for this run; shown at the default 0.5 cutoff.")
