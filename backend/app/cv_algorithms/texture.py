"""
Rotation-invariant circular LBP (P=8, R=1, with bilinear interpolation for the
four diagonal sample points that don't land on the pixel grid) computed on the
image's Hue channel, classified by 1-nearest-neighbor against a bundled
weather-photo training set.

Ported from Homework 7 (ECE 661). The LBP math (bilinear sampling, minimum
circular rotation for rotation invariance, run-length uniform-pattern
encoding) is unchanged, but reworked into vectorized numpy instead of the
original's Python double-loop-per-pixel plus a `BitVector` object per pixel:
the four diagonal samples all land at 45/135/225/315 degrees, which turns out
to always use the *same* bilinear weights (only which four neighbors they
blend differs), so they're computed as plain array arithmetic; and the
"rotate to minimum, then run-length encode" step -- inherently a per-pixel
categorical decision -- is replaced by a 256-entry lookup table (every
possible 8-bit neighbor pattern -> its encoding, computed once at import time)
applied with a single fancy-indexing lookup. Same output, no more per-pixel
Python. A 64x64 image that used to take a couple of seconds now takes under a
millisecond, which matters here because classifying one upload means also
comparing it against several hundred bundled training images.

The deep VGG/ResNet Gram-matrix path from the original is not ported here; it
needs a pretrained network and is a much heavier lift for comparatively little
classification benefit over LBP on this dataset. And the classifier itself
(1-NN over histogram-intersection distance) isn't from the original main.py
-- that script only extracted and saved features; whatever classifier
produced the assignment's confusion matrices lived elsewhere and wasn't
available to port.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

CLASSES = ["cloudy", "rain", "shine", "sunrise"]
TRAINING_DIR = Path(__file__).resolve().parents[2] / "data" / "texture" / "training"
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "texture" / "samples"
SAMPLE_IMAGES = [SAMPLE_DIR / f"{cls}.jpg" for cls in CLASSES]

P = 8


def rgb_to_hue(img_bgr: np.ndarray) -> np.ndarray:
    """Hue channel in degrees [0, 360), matching the original RGB_to_H."""
    img = img_bgr.astype(np.float64) / 255.0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    c = maxc - minc

    hue = np.zeros_like(maxc)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_max = maxc == r
        g_max = (maxc == g) & ~r_max
        b_max = (maxc == b) & ~r_max & ~g_max
        hue[r_max] = 60 * (((g[r_max] - b[r_max]) / c[r_max]) % 6)
        hue[g_max] = 60 * (((b[g_max] - r[g_max]) / c[g_max]) + 2)
        hue[b_max] = 60 * (((r[b_max] - g[b_max]) / c[b_max]) + 4)
    hue[c == 0] = 0
    return np.nan_to_num(hue)


def _min_rotation_bits(bits: str) -> str:
    n = len(bits)
    return min((bits[i:] + bits[:i] for i in range(n)), key=lambda s: int(s, 2))


def _runs(bits: str) -> list[str]:
    runs, i = [], 0
    while i < len(bits):
        j = i
        while j < len(bits) and bits[j] == bits[i]:
            j += 1
        runs.append(bits[i:j])
        i = j
    return runs


def _lbp_encoding(bits: str) -> int:
    min_bits = _min_rotation_bits(bits)
    runs = _runs(min_bits)
    if len(runs) > 2:
        return P + 1
    if len(runs) == 2:
        return len(runs[1])
    return P if runs[0][0] == "1" else 0


# Every possible 8-bit neighbor pattern -> its rotation-invariant uniform-LBP
# encoding (0..9), computed once so per-pixel classification is a lookup.
_ENCODING_LUT = np.array([_lbp_encoding(format(v, "08b")) for v in range(256)])

# Diagonal samples (at 45/135/225/315 degrees, radius 1) fall strictly inside
# the pixel grid, so they're bilinearly interpolated from the center and the
# two flanking axis-aligned neighbors plus the diagonal neighbor itself. Since
# all four are odd multiples of 45 degrees, |cos| = |sin| for each of them, so
# the four weights below are identical regardless of which diagonal it is.
_R = np.cos(np.pi / 4)
_W_CENTER = (1 - _R) ** 2
_W_ADJ = (1 - _R) * _R
_W_DIAG = _R ** 2


def lbp_encoding_map(channel: np.ndarray) -> np.ndarray:
    """channel: single-channel float image (the Hue channel in the original).
    Returns the per-pixel rotation-invariant encoding (0..9), one smaller in
    each dimension than the input (no encoding is defined at the border)."""
    h, w = channel.shape
    c = channel[1:h - 1, 1:w - 1]
    S = channel[2:h, 1:w - 1]
    SE = channel[2:h, 2:w]
    E = channel[1:h - 1, 2:w]
    NE = channel[0:h - 2, 2:w]
    N = channel[0:h - 2, 1:w - 1]
    NW = channel[0:h - 2, 0:w - 2]
    W = channel[1:h - 1, 0:w - 2]
    SW = channel[2:h, 0:w - 2]

    se_i = _W_CENTER * c + _W_ADJ * E + _W_ADJ * S + _W_DIAG * SE
    ne_i = _W_CENTER * c + _W_ADJ * N + _W_ADJ * E + _W_DIAG * NE
    nw_i = _W_CENTER * c + _W_ADJ * W + _W_ADJ * N + _W_DIAG * NW
    sw_i = _W_CENTER * c + _W_ADJ * S + _W_ADJ * W + _W_DIAG * SW

    # p = 0..7, matching the original's S, SE, E, NE, N, NW, W, SW ordering
    samples = [S, se_i, E, ne_i, N, nw_i, W, sw_i]
    pattern = np.zeros(c.shape, dtype=np.uint16)
    for p, val in enumerate(samples):
        bit = (val >= c - 1e-6).astype(np.uint16)
        pattern |= bit << (P - 1 - p)

    return _ENCODING_LUT[pattern]


def lbp_histogram(channel: np.ndarray) -> np.ndarray:
    encodings = lbp_encoding_map(channel)
    hist = np.bincount(((encodings - 1) % (P + 2)).ravel(), minlength=P + 2)
    return hist.astype(np.float64)


def feature_vector(img_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(img_bgr, (64, 64))
    hue = rgb_to_hue(resized)
    hist = lbp_histogram(hue)
    total = hist.sum()
    return hist / total if total > 0 else hist


# Distinct colors (BGR) for each of the 10 rotation-invariant LBP encodings,
# purely for visualization -- 0..7 are "uniform" patterns with that many 1-bits
# in their run, 8 is "all foreground", 9 is "non-uniform" (noisy/high-frequency).
_ENCODING_COLORS = np.array([
    [40, 40, 40], [180, 60, 40], [180, 100, 40], [160, 150, 30], [80, 170, 40],
    [40, 160, 120], [40, 130, 180], [60, 70, 200], [140, 60, 190], [230, 230, 230],
], dtype=np.uint8)


def hue_to_visual(hue_deg: np.ndarray) -> np.ndarray:
    """Colorize a Hue-in-degrees channel as a full-saturation HSV wheel."""
    h8 = np.clip(hue_deg / 2.0, 0, 179).astype(np.uint8)  # OpenCV hue range is [0,179]
    hsv = np.stack([h8, np.full_like(h8, 255), np.full_like(h8, 255)], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def encoding_map_to_visual(encodings: np.ndarray, upscale_to: tuple[int, int] | None = None) -> np.ndarray:
    visual = _ENCODING_COLORS[encodings]
    if upscale_to:
        visual = cv2.resize(visual, upscale_to, interpolation=cv2.INTER_NEAREST)
    return visual


_training_bank: dict | None = None


def _load_training_bank() -> dict:
    global _training_bank
    if _training_bank is not None:
        return _training_bank

    features, labels, paths = [], [], []
    for path in sorted(TRAINING_DIR.glob("*.jpg")):
        name = path.name.lower()
        label = next((c for c in CLASSES if c in name), None)
        if label is None:
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        features.append(feature_vector(img))
        labels.append(CLASSES.index(label))
        paths.append(str(path))

    _training_bank = {
        "features": np.array(features),
        "labels": np.array(labels),
        "paths": paths,
    }
    return _training_bank


def _histogram_intersection_distance(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    # sum of element-wise minimums; converted to a distance (lower = closer)
    return 1 - np.minimum(query[None, :], bank).sum(axis=1)


def classify(img_bgr: np.ndarray, k: int = 5):
    bank = _load_training_bank()

    resized = cv2.resize(img_bgr, (64, 64))
    hue = rgb_to_hue(resized)
    encodings = lbp_encoding_map(hue)
    hist = lbp_histogram(hue)
    total = hist.sum()
    query = hist / total if total > 0 else hist

    dists = _histogram_intersection_distance(query, bank["features"])
    nearest_idx = np.argsort(dists)[:k]

    neighbor_labels = bank["labels"][nearest_idx]
    counts = np.bincount(neighbor_labels, minlength=len(CLASSES))
    predicted = int(np.argmax(counts))

    confidences = counts / counts.sum()
    nearest = [
        {"path": bank["paths"][i], "label": CLASSES[bank["labels"][i]], "distance": float(dists[i])}
        for i in nearest_idx
    ]
    return {
        "predicted": CLASSES[predicted],
        "confidences": {c: float(confidences[i]) for i, c in enumerate(CLASSES)},
        "nearest": nearest,
        "resized_input": resized,
        "hue_visual": hue_to_visual(hue),
        "encoding_visual": encoding_map_to_visual(encodings, upscale_to=(256, 256)),
        "histogram": query.tolist(),
    }
