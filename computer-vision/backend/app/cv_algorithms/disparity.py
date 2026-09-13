"""
Dense stereo disparity via windowed census-transform matching (Task 3 of
Homework 9, ECE 661): for each pixel, encode its MxM neighborhood as a bit
vector of "brighter than center" comparisons (a census transform), then for
each candidate disparity d, score the match between left and right windows by
Hamming distance and keep the d that minimizes it.

The original was a quadruple-nested Python loop (row, col, disparity, window)
that took minutes per window size even on small Middlebury test images. This
port keeps the exact same algorithm but computes the census transform and the
per-disparity Hamming cost as vectorized numpy array operations, so a demo
upload resolves in a couple of seconds.
"""
from __future__ import annotations

import numpy as np
import cv2

MAX_DIM = 420  # stereo matching is O(d_max * H * W * M^2); keep uploads small


def resize_for_demo(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _census_transform(img: np.ndarray, window: int) -> np.ndarray:
    """Boolean array of shape (H, W, window*window): is neighbor brighter than center."""
    r = window // 2
    padded = np.pad(img, r, mode="constant", constant_values=0).astype(np.int16)
    h, w = img.shape
    center = img.astype(np.int16)
    bits = np.empty((h, w, window * window), dtype=bool)
    idx = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            bits[:, :, idx] = padded[r + dy:r + dy + h, r + dx:r + dx + w] > center
            idx += 1
    return bits


def compute_disparity(img_left_gray: np.ndarray, img_right_gray: np.ndarray,
                       window: int, d_max: int, progress=None) -> np.ndarray:
    r = window // 2
    h, w = img_left_gray.shape
    census_left = _census_transform(img_left_gray, window)
    census_right = _census_transform(img_right_gray, window)

    d_max = min(d_max, w - 1)
    cost_volume = np.full((d_max + 1, h, w), np.inf, dtype=np.float32)

    for d in range(d_max + 1):
        if progress:
            progress(d, d_max)
        shifted = np.zeros_like(census_right)
        if d == 0:
            shifted = census_right
        else:
            shifted[:, d:, :] = census_right[:, :-d, :]
        cost = np.count_nonzero(census_left ^ shifted, axis=-1).astype(np.float32)
        cost_volume[d] = cost
        # a pixel only has a valid candidate at disparity d if its right-image
        # window (j - d - r) stays in bounds, matching the original's
        # `range(min(j - M//2, d_max))` search bound
        invalid_cols = d + r
        if invalid_cols > 0:
            cost_volume[d, :, :invalid_cols] = np.inf

    disparity = np.argmin(cost_volume, axis=0).astype(np.float32)
    all_inf = np.all(np.isinf(cost_volume), axis=0)
    disparity[all_inf] = 0
    return disparity


def disparity_to_visual(disparity: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    gray_u8 = normalized.astype(np.uint8)
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)


POINT_CLOUD_MAX_DIM = 240  # downsample target for an interactive point count


def build_point_cloud(disparity: np.ndarray, left_bgr: np.ndarray, max_dim: int = POINT_CLOUD_MAX_DIM):
    """Unproject the disparity map into a colored 3D point cloud.

    There's no real camera calibration for an arbitrary uploaded stereo pair
    (no known focal length or baseline), so this uses a nominal pinhole model
    (focal length as a fraction of image width, an arbitrary baseline) --
    it's relative/uncalibrated depth, not metric. Good enough to *look*
    correct (near things bulge toward the viewer, a slanted surface looks
    slanted) when spun around, not to measure real-world distances from.
    """
    h, w = disparity.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        disp_small = cv2.resize(disparity, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        color_small = cv2.resize(left_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        disp_small, color_small = disparity, left_bgr
    h2, w2 = disp_small.shape

    min_disp = max(1.0, np.percentile(disp_small[disp_small > 0], 2)) if np.any(disp_small > 0) else 1.0
    valid = disp_small > min_disp
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    d = disp_small[ys, xs].astype(np.float64)

    focal = 0.9 * w2
    baseline = 60.0
    Z = focal * baseline / d

    z_lo, z_hi = np.percentile(Z, [1, 97])
    keep = (Z >= z_lo) & (Z <= z_hi)
    xs, ys, Z = xs[keep], ys[keep], Z[keep]

    cx, cy = w2 / 2.0, h2 / 2.0
    X = (xs - cx) * Z / focal
    Y = -(ys - cy) * Z / focal  # flip: image rows increase downward, world Y should increase upward

    positions = np.stack([X, Y, -Z], axis=1).astype(np.float32)
    colors_bgr = color_small[ys, xs]
    colors_rgb = colors_bgr[:, ::-1].astype(np.uint8)
    return positions, colors_rgb


def run_disparity(img_left_bgr: np.ndarray, img_right_bgr: np.ndarray, window: int, d_max: int, progress=None):
    left = resize_for_demo(img_left_bgr)
    right = resize_for_demo(img_right_bgr)
    if left.shape[:2] != right.shape[:2]:
        h = min(left.shape[0], right.shape[0])
        w = min(left.shape[1], right.shape[1])
        left = cv2.resize(left, (w, h))
        right = cv2.resize(right, (w, h))

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disparity = compute_disparity(gray_left, gray_right, window, d_max, progress=progress)
    normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    grayscale = normalized.astype(np.uint8)
    visual = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
    positions, colors = build_point_cloud(disparity, left)
    return {
        "input_left_resized": left,
        "input_right_resized": right,
        "disparity_gray": grayscale,
        "disparity_color": visual,
        "point_positions": positions,
        "point_colors": colors,
    }
