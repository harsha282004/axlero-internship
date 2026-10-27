"""Binary classification metrics and validation-only threshold selection.

Convention: positive class = PNEUMONIA (1). A sample is predicted positive when
probability >= threshold. Undefined metrics (e.g. a class is absent) are None.
"""
import logging

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             roc_auc_score)

from src.utils import CLASS_NAMES

log = logging.getLogger("cxr")

THRESHOLD_STRATEGIES = ("youden", "f1", "sensitivity", "fixed")
MAX_CANDIDATES = 2000


def _div(a, b):
    return float(a / b) if b else None


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise ValueError(f"y_true and y_prob must be 1-D and equal length, got "
                         f"{y_true.shape} vs {y_prob.shape}")
    if len(y_true) == 0:
        raise ValueError("Cannot compute metrics on an empty set")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = _div(tp, tp + fn)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)

    has_both = len(np.unique(y_true)) == 2
    if not has_both:
        log.warning("Only one class present in y_true; ROC-AUC and PR-AUC are undefined")
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": sens,  # recall of PNEUMONIA
        "specificity": _div(tn, tn + fp),
        "precision": prec,
        "f1": f1,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],  # rows=true, cols=pred
        "confusion_matrix_labels": CLASS_NAMES,
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0),
    }


def _candidate_thresholds(y_prob):
    u = np.unique(y_prob)
    if len(u) > MAX_CANDIDATES:
        u = np.unique(np.quantile(y_prob, np.linspace(0, 1, MAX_CANDIDATES)))
    mids = (u[:-1] + u[1:]) / 2 if len(u) > 1 else np.array([])
    return np.concatenate([[u[0] - 1e-6], mids, [u[-1] + 1e-6]])


def select_threshold(y_true, y_prob, strategy="youden", target_sensitivity=0.95):
    """Pick a decision threshold. Call ONLY with validation data.

    youden:      maximise sensitivity + specificity - 1
    f1:          maximise F1 of PNEUMONIA
    sensitivity: highest specificity among thresholds with sensitivity >= target
    fixed:       0.5
    Returns (threshold, info dict). Ties go to the threshold closest to 0.5.
    """
    if strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {THRESHOLD_STRATEGIES}")
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if strategy == "fixed":
        return 0.5, {"strategy": "fixed"}
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        log.warning("Validation data lacks a class; falling back to threshold 0.5")
        return 0.5, {"strategy": "fixed", "note": "fallback: single class in validation data"}

    cands = _candidate_thresholds(y_prob)
    pred = y_prob[None, :] >= cands[:, None]
    tp = (pred & (y_true == 1)).sum(axis=1)
    fp = (pred & (y_true == 0)).sum(axis=1)
    sens = tp / n_pos
    spec = 1 - fp / n_neg
    f1 = 2 * tp / np.maximum(2 * tp + fp + (n_pos - tp), 1)

    if strategy == "youden":
        score = sens + spec - 1
    elif strategy == "f1":
        score = f1
    else:
        score = np.where(sens >= target_sensitivity, spec, -np.inf)
        if np.isinf(score).all():  # cannot happen when n_pos > 0, kept as a guard
            score = sens
    best = score.max()
    idx = np.flatnonzero(np.isclose(score, best))
    i = idx[np.argmin(np.abs(cands[idx] - 0.5))]
    info = {"strategy": strategy, "val_sensitivity": float(sens[i]), "val_specificity": float(spec[i]),
            "val_f1": float(f1[i])}
    if strategy == "sensitivity":
        info["target_sensitivity"] = target_sensitivity
    return float(cands[i]), info
