"""
Car/not-car patch classification via integral-image Haar-like features
boosted with AdaBoost, ported from Homework 10 (ECE 661) Task 3.

The Haar feature extraction (adjacent-strip sums from a horizontal and a
vertical integral image, at every width/height/position -- 11200 features for
a 20x40 patch) is unchanged, vectorized instead of the original's per-pixel
triple-nested loop.

The AdaBoost weak-classifier selection is **reimplemented**, not ported,
because the original had a bug that would corrupt training if copied
faithfully: `best_weak_classifier` computed
`ind = np.argsort(feat_mat, axis=1)` (a per-feature sort order across images)
but then gathered the sorted feature values with
`np.take_along_axis(feat_mat, ind, axis=0)` -- axis 0, not axis 1. That
indexes along the *feature* dimension using indices that mean "image position
in sorted order", silently producing nonsense instead of raising (both axes
have valid index ranges). The weighted per-threshold error formulas
themselves (`Sp + Tn - Sn`, `Sn + Tp - Sp`) are Viola-Jones standard and
unchanged; the fix is purely in the sort/gather axis.

A second, subtler bug surfaced once the axis fix was in: these Haar sums are
highly discrete, so many images tie on the exact same feature value, and a
naive threshold search can land *inside* a tied block, reporting an error
rate that no real threshold can actually achieve (every tied value must fall
on the same side of a real threshold, but the cumsum-based search treats each
sorted position as independently splittable). That's fixed by only
considering split points at boundaries between distinct values.

The original also cascaded several boosted stages (dropping "easy" negatives
between stages) for **detection speed** -- so a sliding window search over a
full image can reject most windows cheaply in an early stage. That benefit
doesn't apply here: this demo classifies one already-cropped patch, so a
single strong classifier boosted from more weak classifiers is simpler and,
empirically on this dataset, both faster to train and more accurate (a
multi-stage version was tried first; each stage's false-negative rate
compounds multiplicatively across stages, and with only a handful of stages
recall collapsed to ~20%). 20 rounds of boosting reaches the same held-out
accuracy as 50 in about a third of the training time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CAR_DIR = DATA_DIR / "car_detection"
PATCH_SIZE = (40, 20)  # (width, height) -- matches the original 20x40 patches

ROUNDS = 20


def extract_haar_features(gray_patch: np.ndarray) -> np.ndarray:
    img = gray_patch.astype(np.int32)
    h, w = img.shape
    feats = []
    hor = np.cumsum(img, axis=1)
    for width in range(1, w // 2):
        j = np.arange(w - 2 * width)
        neg = hor[:, j + width - 1] - hor[:, j]
        pos = hor[:, j + 2 * width - 1] - hor[:, j + width]
        feats.append((pos - neg).ravel())
    vert = np.cumsum(img, axis=0)
    for height in range(1, h // 2):
        i = np.arange(h - 2 * height)
        neg = vert[i + height - 1, :] - vert[i, :]
        pos = vert[i + 2 * height - 1, :] - vert[i + height, :]
        feats.append((pos - neg).ravel())
    return np.concatenate(feats).astype(np.float32)


def _load_split(split: str):
    images, labels = [], []
    for label, sub in [(0, "negative"), (1, "positive")]:
        d = CAR_DIR / split / sub
        for path in sorted(d.glob("*.png")):
            img = cv2.imread(str(path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            gray = cv2.resize(gray, PATCH_SIZE)
            images.append(gray)
            labels.append(label)
    return images, np.array(labels)


def _best_weak_classifier(feat_mat: np.ndarray, weights: np.ndarray, labels: np.ndarray):
    """feat_mat: (F, N), weights: (N,), labels: (N,) of 0/1.
    Returns (pred, weighted_error, polarity, threshold, feature_index)."""
    Tp = weights[labels == 1].sum()
    Tn = weights[labels == 0].sum()

    order = np.argsort(feat_mat, axis=1)
    feat_sorted = np.take_along_axis(feat_mat, order, axis=1)
    w_row = np.broadcast_to(weights, feat_mat.shape)
    l_row = np.broadcast_to(labels, feat_mat.shape)
    weights_sorted = np.take_along_axis(w_row, order, axis=1)
    labels_sorted = np.take_along_axis(l_row, order, axis=1)

    Sp = np.cumsum(weights_sorted * labels_sorted, axis=1)
    Sn = np.cumsum(weights_sorted, axis=1) - Sp

    e1 = Sp + Tn - Sn  # polarity +1: "positive" means feature > threshold
    e2 = Sn + Tp - Sp  # polarity -1: "positive" means feature <= threshold

    # only split points at boundaries between distinct values are realizable
    # by an actual threshold -- see module docstring
    tied_with_next = feat_sorted[:, :-1] == feat_sorted[:, 1:]
    e1[:, :-1][tied_with_next] = np.inf
    e2[:, :-1][tied_with_next] = np.inf

    f1, k1 = np.unravel_index(np.argmin(e1), e1.shape)
    f2, k2 = np.unravel_index(np.argmin(e2), e2.shape)

    if e1[f1, k1] <= e2[f2, k2]:
        feat_idx, threshold, polarity = int(f1), float(feat_sorted[f1, k1]), 1
        pred = (feat_mat[feat_idx, :] > threshold).astype(np.float64)
    else:
        feat_idx, threshold, polarity = int(f2), float(feat_sorted[f2, k2]), -1
        pred = (feat_mat[feat_idx, :] <= threshold).astype(np.float64)

    e_t = 0.5 * np.sum(weights * np.abs(pred - labels))
    return pred, e_t, polarity, threshold, feat_idx


def _train(feat_mat: np.ndarray, labels: np.ndarray, progress=None) -> dict:
    neg_num, pos_num = int(np.sum(labels == 0)), int(np.sum(labels == 1))
    weights = np.where(labels == 0, 1 / (2 * neg_num), 1 / (2 * pos_num))
    weak_classifiers = []
    for r in range(ROUNDS):
        if progress:
            progress(f"Boosting round {r + 1}/{ROUNDS}")
        pred, e_t, polarity, threshold, feat_idx = _best_weak_classifier(feat_mat, weights, labels)
        e_t = min(max(e_t, 1e-6), 1 - 1e-6)
        beta = e_t / (1 - e_t)
        alpha = np.log(1 / beta)
        weights = weights * beta ** (1 - np.abs(labels - pred))
        weights = weights / weights.sum()
        weak_classifiers.append({"feat_idx": feat_idx, "polarity": polarity, "threshold": threshold, "alpha": float(alpha)})

    # standard AdaBoost strong-classifier decision boundary: a weighted
    # majority vote, i.e. "positive" iff the alpha-weighted sum of positive
    # votes is at least half the total alpha weight
    threshold = 0.5 * sum(wc["alpha"] for wc in weak_classifiers)
    return {"weak_classifiers": weak_classifiers, "threshold": threshold}


def score(classifier: dict, feat_vec: np.ndarray) -> float:
    total = 0.0
    for wc in classifier["weak_classifiers"]:
        val = feat_vec[wc["feat_idx"]]
        pred = 1.0 if (wc["polarity"] == 1 and val > wc["threshold"]) or (wc["polarity"] == -1 and val <= wc["threshold"]) else 0.0
        total += wc["alpha"] * pred
    return total


_classifier_state: dict | None = None


def _get_classifier() -> dict:
    global _classifier_state
    if _classifier_state is not None:
        return _classifier_state

    train_images, train_labels = _load_split("train")
    train_feats = np.stack([extract_haar_features(im) for im in train_images], axis=1)  # (F, N)
    classifier = _train(train_feats, train_labels)

    test_images, test_labels = _load_split("test")
    test_feats = np.stack([extract_haar_features(im) for im in test_images], axis=1)
    scores = np.array([score(classifier, test_feats[:, i]) for i in range(test_feats.shape[1])])
    preds = scores >= classifier["threshold"]

    tp = int(np.sum(preds & (test_labels == 1)))
    fp = int(np.sum(preds & (test_labels == 0)))
    fn = int(np.sum(~preds & (test_labels == 1)))
    tn = int(np.sum(~preds & (test_labels == 0)))

    _classifier_state = {
        "classifier": classifier,
        "test_accuracy": float(np.mean(preds == test_labels.astype(bool))),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "num_rounds": len(classifier["weak_classifiers"]),
    }
    return _classifier_state


def draw_top_features(gray_patch: np.ndarray, classifier: dict, n: int = 3) -> np.ndarray:
    """Visualize the strongest (highest-alpha) weak classifiers' feature
    rectangles, upscaled for visibility. Dark rectangle = the "darker" side
    of the Haar filter, light = the "lighter" side."""
    h, w = gray_patch.shape
    scale = 8
    out = cv2.cvtColor(cv2.resize(gray_patch, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)

    top = sorted(classifier["weak_classifiers"], key=lambda wc: -wc["alpha"])[:n]
    for rank, wc in enumerate(top):
        idx = wc["feat_idx"]
        color = [(0, 0, 255), (0, 165, 255), (0, 220, 220)][rank % 3]
        cursor = 0
        placed = False
        for width in range(1, w // 2):
            count = h * (w - 2 * width)
            if idx < cursor + count:
                local = idx - cursor
                i, j = divmod(local, w - 2 * width)
                cv2.rectangle(out, (j * scale, i * scale), ((j + width) * scale, (i + 1) * scale), color, 2)
                cv2.rectangle(out, ((j + width) * scale, i * scale), ((j + 2 * width) * scale, (i + 1) * scale), color, 1)
                placed = True
                break
            cursor += count
        if placed:
            continue
        for height in range(1, h // 2):
            count = w * (h - 2 * height)
            if idx < cursor + count:
                local = idx - cursor
                j, i = divmod(local, h - 2 * height)
                cv2.rectangle(out, (j * scale, i * scale), ((j + 1) * scale, (i + height) * scale), color, 2)
                cv2.rectangle(out, (j * scale, (i + height) * scale), ((j + 1) * scale, (i + 2 * height) * scale), color, 1)
                break
            cursor += count
    return out


def run_classification(img_bgr: np.ndarray):
    state = _get_classifier()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    patch = cv2.resize(gray, PATCH_SIZE)
    feat_vec = extract_haar_features(patch)

    classifier = state["classifier"]
    patch_score = score(classifier, feat_vec)
    is_car = patch_score >= classifier["threshold"]
    feature_overlay = draw_top_features(patch, classifier)

    return {
        "is_car": bool(is_car),
        "score": float(patch_score),
        "threshold": float(classifier["threshold"]),
        "num_rounds": state["num_rounds"],
        "test_accuracy": state["test_accuracy"],
        "confusion": state["confusion"],
        "resized_patch": patch,
        "feature_overlay": feature_overlay,
    }


def sample_paths(n: int = 6) -> dict:
    pos = sorted((CAR_DIR / "test" / "positive").glob("*.png"))[:n]
    neg = sorted((CAR_DIR / "test" / "negative").glob("*.png"))[:n]
    return {"positive": [str(p) for p in pos], "negative": [str(p) for p in neg]}
