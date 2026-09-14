"""
Eigenfaces (PCA) and Fisherfaces (Fisher-LDA) face recognition, ported from
Homework 10 (ECE 661) Task 1. Both subspace projections and the 1-NN
classifier are numerically unchanged from the original -- the math was
already vectorized numpy, not per-pixel loops. What's added is a small
inference layer: caching the trained bases (they're nested, so any K up to a
fixed max can be sliced out of one cached basis instead of retraining per
request) and a K-sweep accuracy curve.

This is closed-set recognition against exactly the 30 subjects in the bundled
training set, with no face detection/alignment step (matching the original) --
an upload gets resized to 128x128 and its raw pixel intensities are projected
directly, so it will always return whichever of those 30 people is the
nearest match in eigenspace/fisherspace, however good or bad that match
actually is. It is sensitive to cropping/pose/lighting for exactly that reason.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

FACE_DIR = Path(__file__).resolve().parents[2] / "data" / "face_recognition"
IMG_SIZE = (128, 128)
MAX_K = 29  # LDA's hard cap is (num_classes - 1); used for PCA too for a shared slider


def _load_split(split: str):
    img_dir = FACE_DIR / split
    labels, vecs, paths = [], [], []
    for path in sorted(img_dir.glob("*.png")):
        label = int(path.name[:2])
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        vec = img.flatten().astype(np.float64)
        vecs.append(vec / np.linalg.norm(vec))
        labels.append(label)
        paths.append(str(path))
    return np.array(labels), np.array(vecs).T, paths  # vecs: (D, N)


def _normalize_cols(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=0, keepdims=True)


def _pca_basis(x: np.ndarray, mean: np.ndarray, k_max: int) -> np.ndarray:
    xc = x - mean
    _, _, vh = np.linalg.svd(xc.T @ xc)
    w = _normalize_cols(xc @ vh)
    return w[:, :k_max]


def _lda_basis(labels: np.ndarray, x: np.ndarray, mean: np.ndarray, k_max: int) -> np.ndarray:
    classes = np.unique(labels)
    c = len(classes)
    d = x.shape[0]
    m_i = np.zeros((d, c))
    x_within = np.zeros_like(x)
    for idx, cls in enumerate(classes):
        mask = labels == cls
        m_i[:, idx] = x[:, mask].mean(axis=1)
        x_within[:, mask] = x[:, mask] - m_i[:, idx:idx + 1]

    m_between = m_i - mean
    _, eigvals, vh1 = np.linalg.svd(m_between.T @ m_between)
    v = _normalize_cols(m_between @ vh1)
    y = v[:, :c - 1]  # the c-th eigenvalue is ~0 (mean-centered class means have rank c-1)
    d_b = np.diag(eigvals[:c - 1] + 1e-8)
    z = y @ (np.linalg.inv(d_b) ** 0.5)
    _, _, vh2 = np.linalg.svd((z.T @ x_within) @ (z.T @ x_within).T)
    u = _normalize_cols(z @ vh2)
    return u[:, : min(k_max, c - 1)]


_state: dict | None = None


def _get_state() -> dict:
    global _state
    if _state is not None:
        return _state

    train_labels, train_vecs, train_paths = _load_split("train")
    test_labels, test_vecs, test_paths = _load_split("test")
    mean = train_vecs.mean(axis=1, keepdims=True)

    _state = {
        "train_labels": train_labels, "train_vecs": train_vecs, "train_paths": train_paths,
        "test_labels": test_labels, "test_vecs": test_vecs, "test_paths": test_paths,
        "mean": mean,
        "pca_basis": _pca_basis(train_vecs, mean, MAX_K),
        "lda_basis": _lda_basis(train_labels, train_vecs, mean, MAX_K),
    }
    return _state


def _nearest_neighbor(train_feats: np.ndarray, train_labels: np.ndarray, train_paths, query_feat: np.ndarray):
    dist = np.linalg.norm(train_feats - query_feat.reshape(-1, 1), axis=0)
    idx = int(np.argmin(dist))
    return {"label": int(train_labels[idx]), "path": train_paths[idx], "distance": float(dist[idx])}


def _to_feature_vector(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    resized = cv2.resize(gray, IMG_SIZE)
    vec = resized.flatten().astype(np.float64)
    return vec / np.linalg.norm(vec)


def _basis_column_to_image(column: np.ndarray) -> np.ndarray:
    """A basis vector is a direction in pixel space, not an image -- normalize
    its (arbitrary sign, arbitrary scale) values to 0..255 purely for display."""
    img = column.reshape(IMG_SIZE)
    lo, hi = img.min(), img.max()
    normalized = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return (normalized * 255).astype(np.uint8)


def basis_thumbnails(basis: np.ndarray, n: int) -> list[np.ndarray]:
    return [_basis_column_to_image(basis[:, i]) for i in range(min(n, basis.shape[1]))]


def classify(img_bgr: np.ndarray, k: int):
    state = _get_state()
    k = max(1, min(k, MAX_K))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    resized_input = cv2.resize(gray, IMG_SIZE)
    query = _to_feature_vector(img_bgr).reshape(-1, 1)
    centered = query - state["mean"]

    pca_w = state["pca_basis"][:, :k]
    lda_w = state["lda_basis"][:, : min(k, state["lda_basis"].shape[1])]

    pca_train_feats = pca_w.T @ (state["train_vecs"] - state["mean"])
    lda_train_feats = lda_w.T @ (state["train_vecs"] - state["mean"])

    pca_match = _nearest_neighbor(pca_train_feats, state["train_labels"], state["train_paths"], pca_w.T @ centered)
    lda_match = _nearest_neighbor(lda_train_feats, state["train_labels"], state["train_paths"], lda_w.T @ centered)
    return {
        "pca": pca_match,
        "lda": lda_match,
        "resized_input": resized_input,
        "eigenfaces": basis_thumbnails(state["pca_basis"], 6),
        "fisherfaces": basis_thumbnails(state["lda_basis"], 6),
    }


def _accuracy_at_k(k: int) -> dict:
    state = _get_state()
    pca_w = state["pca_basis"][:, :k]
    lda_w = state["lda_basis"][:, : min(k, state["lda_basis"].shape[1])]

    def eval_method(w):
        train_feats = w.T @ (state["train_vecs"] - state["mean"])
        test_feats = w.T @ (state["test_vecs"] - state["mean"])
        dists = np.linalg.norm(train_feats[:, :, None] - test_feats[:, None, :], axis=0)  # (N_train, N_test)
        pred = state["train_labels"][np.argmin(dists, axis=0)]
        return float(np.mean(pred == state["test_labels"]) * 100)

    return {"pca": eval_method(pca_w), "lda": eval_method(lda_w)}


_accuracy_curve_cache: dict | None = None


def accuracy_curve() -> dict:
    global _accuracy_curve_cache
    if _accuracy_curve_cache is not None:
        return _accuracy_curve_cache
    ks = list(range(1, MAX_K + 1))
    pca_acc, lda_acc = [], []
    for k in ks:
        acc = _accuracy_at_k(k)
        pca_acc.append(acc["pca"])
        lda_acc.append(acc["lda"])
    _accuracy_curve_cache = {"k": ks, "pca": pca_acc, "lda": lda_acc}
    return _accuracy_curve_cache


def sample_test_paths(n: int = 8) -> list[str]:
    state = _get_state()
    # one sample per subject, spread across the first n subjects
    seen: dict[int, str] = {}
    for label, path in zip(state["test_labels"], state["test_paths"]):
        if label not in seen:
            seen[label] = path
        if len(seen) >= n:
            break
    return list(seen.values())
