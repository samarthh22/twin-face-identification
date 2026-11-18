"""Validation-only calibration and diagnostics; higher cosine means same identity."""

import numpy as np
from sklearn.metrics import roc_curve, auc


def binary_metrics(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if len(s) != len(y) or not np.isfinite(s).all() or not set(y).issubset({0, 1}):
        raise ValueError("Invalid binary scores/labels")
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if not npos or not nneg:
        return {"supported": False, "positive_count": npos, "negative_count": nneg}
    fpr, tpr, th = roc_curve(y, s, drop_intermediate=False)
    delta = fpr + tpr - 1
    hi = int(np.flatnonzero(delta >= 0)[0])
    lo = max(0, hi - 1)
    w = 0.0 if delta[hi] == delta[lo] else float(-delta[lo] / (delta[hi] - delta[lo]))
    eer = float(fpr[lo] + w * (fpr[hi] - fpr[lo]))
    tar = {
        str(far): (float(tpr[fpr <= far].max()) if nneg * far >= 1 else None)
        for far in (0.01, 0.001)
    }
    return {
        "supported": True,
        "eer": eer,
        "auc": float(auc(fpr, tpr)),
        "tar_at_far": tar,
        "positive_count": npos,
        "negative_count": nneg,
        "far_resolution": 1 / nneg,
        "eer_definition": "linear ROC crossing; not a deployed threshold",
    }
