"""Single-image inference for the X-Ray Prediction page.

Reuses the project's own architecture/preprocessing (src.evaluate.load_model,
src.dataset.build_transforms) so dashboard predictions are produced by the exact same
code path as src.evaluate -- not a reimplementation. CPU-only by default (uploads are
single images; no need to require CUDA for the dashboard).
"""
import io

import torch
from PIL import Image, UnidentifiedImageError

from src.dataset import build_transforms
from src.evaluate import load_model
from src.utils import CLASS_NAMES, get_device


class IncompatibleCheckpointError(Exception):
    pass


class InvalidImageError(Exception):
    pass


def open_uploaded_image(uploaded_file):
    """Decode an uploaded file into a PIL image, held in memory only (never written to disk)."""
    try:
        data = uploaded_file.getvalue()
        im = Image.open(io.BytesIO(data))
        im.load()
        return im
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise InvalidImageError(f"Could not read this file as an image: {e}") from e


def load_checkpoint_for_inference(checkpoint_path, device=None):
    """Load a checkpoint via the project's own loader; wraps any failure as incompatible."""
    device = device or get_device("cpu")
    try:
        model, ckpt = load_model(checkpoint_path, device)
        model.eval()
        return model, ckpt, device
    except Exception as e:  # corrupt file, unknown architecture, shape mismatch, etc.
        raise IncompatibleCheckpointError(
            f"This checkpoint could not be loaded with the project's model loader: {e}") from e


@torch.inference_mode()
def predict_image(model, ckpt, device, pil_image):
    """Run one image through the checkpoint's own saved preprocessing; returns a result dict."""
    preprocess = ckpt["model"].get("preprocess", "resize")
    transform = build_transforms(train=False, preprocess=preprocess)
    x = transform(pil_image).unsqueeze(0).to(device)
    logit = model(x).squeeze(1).float()
    prob_pneumonia = torch.sigmoid(logit).item()
    pred_idx = int(prob_pneumonia >= 0.5)
    return {
        "preprocess": preprocess,
        "prob_pneumonia": prob_pneumonia,
        "prob_normal": 1.0 - prob_pneumonia,
        "predicted_class": CLASS_NAMES[pred_idx],
        "default_threshold_used": 0.5,
        "note": "Prediction at a fixed 0.5 cutoff for display; it is NOT the run's saved, "
                "validation-selected decision threshold (shown separately if available).",
    }
