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

Evaluated against Middlebury ground truth (the `Task3Images` "cones" pair,
which ships with a ground-truth disparity map), this from-scratch matcher
gets ~74-78% of pixels within 2px of the true disparity at window sizes 9-15
-- a legitimate result for pure local window matching, not a broken
implementation. The remaining ~25% isn't spread evenly, though: local
matching has no way to prefer spatially-consistent answers, so it's very
unreliable in flat/textureless regions specifically (many windows there look
almost identical), and a wrong match there doesn't fail by a little -- it
tends to land at a wildly wrong disparity. Unprojected into 3D, a coherent
foreground object (mostly-correct disparities, clustered at the right depth)
sits alongside a "background" that's actually scattered across many wrong
depths, which visually reads as noise overwhelming the object even though
it's a minority of points.

`compute_filtered_disparity` adds the standard fix for exactly this failure
mode: a left-right consistency check (matching left-to-right and
right-to-left independently and keeping only pixels where both agree, which
rejects most occlusion/ambiguous-region errors) plus a match-confidence
threshold on the raw Hamming cost. `sgbm_disparity` is OpenCV's Semi-Global
Matching as a reference point -- unlike per-pixel-independent window
matching, SGM aggregates cost along multiple directions with a penalty for
disparity discontinuities, so it can use context from neighboring pixels
instead of guessing blind in a flat region. On the same ground truth, it
reaches ~94% within 2px.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

MAX_DIM = 420  # stereo matching is O(d_max * H * W * M^2); keep uploads small

# A properly rectified, well-textured, Middlebury-quality stereo pair
# ("cones") -- bundled because a *good* result depends entirely on the input
# being a genuinely rectified pair (matching points on the same row) with a
# pure horizontal camera shift, which casual handheld photos almost never
# are. This is the pair this module's own accuracy numbers were measured on.
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "disparity"
SAMPLE_LEFT = SAMPLE_DIR / "left.png"
SAMPLE_RIGHT = SAMPLE_DIR / "right.png"


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
    """Left-as-reference census disparity -- the original algorithm, unchanged."""
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


def _cost_volume(census_ref: np.ndarray, census_other: np.ndarray, d_max: int, forward: bool) -> np.ndarray:
    """forward=True: reference pixel j is matched against other pixel (j - d)
    (the left-as-reference convention). forward=False: reference pixel j is
    matched against other pixel (j + d) (right-as-reference)."""
    h, w, _ = census_ref.shape
    cost_volume = np.full((d_max + 1, h, w), np.inf, dtype=np.float32)
    for d in range(d_max + 1):
        shifted = np.zeros_like(census_other)
        if d == 0:
            shifted = census_other
        elif forward:
            shifted[:, d:, :] = census_other[:, :-d, :]
        else:
            shifted[:, :-d, :] = census_other[:, d:, :]
        cost_volume[d] = np.count_nonzero(census_ref ^ shifted, axis=-1).astype(np.float32)
        if d > 0:
            if forward:
                cost_volume[d, :, :d] = np.inf
            else:
                cost_volume[d, :, -d:] = np.inf
    return cost_volume


def compute_filtered_disparity(img_left_gray: np.ndarray, img_right_gray: np.ndarray, window: int, d_max: int,
                                lr_tolerance: float = 1.0, max_cost_frac: float = 0.3):
    """Census disparity with the standard reliability filters: match left-to-right
    and right-to-left independently and keep only pixels where they agree
    (rejects most occlusion/ambiguous-region errors), plus a raw match-cost
    threshold (rejects low-confidence matches even when they happen to agree).
    Returns (disparity, valid_mask)."""
    h, w = img_left_gray.shape
    d_max = min(d_max, w - 1)
    census_left = _census_transform(img_left_gray, window)
    census_right = _census_transform(img_right_gray, window)

    cost_l = _cost_volume(census_left, census_right, d_max, forward=True)
    cost_r = _cost_volume(census_right, census_left, d_max, forward=False)

    disp_left = np.argmin(cost_l, axis=0).astype(np.int32)
    min_cost_left = np.min(cost_l, axis=0)
    disp_right = np.argmin(cost_r, axis=0).astype(np.int32)

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    x_match = xs - disp_left
    in_range = (x_match >= 0) & (x_match < w)
    x_match_clipped = np.clip(x_match, 0, w - 1)
    d_right_at_match = disp_right[ys, x_match_clipped]
    consistent = in_range & (np.abs(disp_left - d_right_at_match) <= lr_tolerance)

    max_bits = window * window
    confident = (min_cost_left / max_bits) <= max_cost_frac

    valid = consistent & confident
    return disp_left.astype(np.float32), valid


def sgbm_disparity(img_left_gray: np.ndarray, img_right_gray: np.ndarray, window: int, d_max: int):
    """OpenCV's Semi-Global Matching, as a reference point: instead of scoring
    every pixel's window independently, it aggregates matching cost along
    several directions with a penalty for neighboring disparities that
    disagree, so a flat/textureless patch inherits a plausible answer from
    its surroundings instead of picking whichever noisy window happens to
    score lowest. Returns (disparity, valid_mask)."""
    num_disparities = max(16, ((d_max // 16) + 1) * 16)
    block_size = max(3, window | 1)  # StereoSGBM wants an odd block size
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = stereo.compute(img_left_gray, img_right_gray).astype(np.float32) / 16.0
    valid = raw >= 0
    disparity = np.where(valid, raw, 0)
    return disparity, valid


def disparity_to_visual(disparity: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    ref = disparity[valid] if valid is not None and valid.any() else disparity
    lo, hi = float(ref.min()), float(ref.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0
    normalized = np.clip((disparity - lo) / (hi - lo), 0, 1)
    gray_u8 = (normalized * 255).astype(np.uint8)
    visual = cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)
    if valid is not None:
        visual[~valid] = (30, 30, 30)
    return visual


POINT_CLOUD_MAX_DIM = 240  # downsample target for an interactive point count


def build_point_cloud(disparity: np.ndarray, left_bgr: np.ndarray, valid_mask: np.ndarray | None = None,
                       max_dim: int = POINT_CLOUD_MAX_DIM):
    """Unproject the disparity map into a colored 3D point cloud.

    There's no real camera calibration for an arbitrary uploaded stereo pair
    (no known focal length or baseline), so this uses a nominal pinhole model
    (focal length as a fraction of image width, an arbitrary baseline) --
    it's relative/uncalibrated depth, not metric. Good enough to *look*
    correct (near things bulge toward the viewer, a slanted surface looks
    slanted) when spun around, not to measure real-world distances from.

    `valid_mask`, when given (from `compute_filtered_disparity` or
    `sgbm_disparity`), excludes unreliable matches *before* they become 3D
    points -- this is what actually fixes a noisy background overwhelming the
    reconstructed structure, rather than just hiding it visually.
    """
    h, w = disparity.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        disp_small = cv2.resize(disparity, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        color_small = cv2.resize(left_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_small = (
            cv2.resize(valid_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0
            if valid_mask is not None else None
        )
    else:
        disp_small, color_small, mask_small = disparity, left_bgr, valid_mask
    h2, w2 = disp_small.shape

    valid = mask_small if mask_small is not None else np.ones_like(disp_small, dtype=bool)
    valid = valid & (disp_small > 0)
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    if mask_small is None:
        # no reliability mask available: fall back to trimming the lowest
        # disparities (most depth-unstable) the way the original heuristic did
        min_disp = max(1.0, np.percentile(disp_small[valid], 2))
        valid = valid & (disp_small > min_disp)

    ys, xs = np.nonzero(valid)
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

    if progress:
        progress("Matching (census transform)")
    disparity = compute_disparity(gray_left, gray_right, window, d_max, progress=progress)
    grayscale = disparity_to_visual(disparity)  # kept as before (no masking) for the raw step
    normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    visual = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_JET)

    if progress:
        progress("Left-right consistency check")
    filtered_disparity, filtered_valid = compute_filtered_disparity(gray_left, gray_right, window, d_max)
    filtered_visual = disparity_to_visual(filtered_disparity, filtered_valid)

    if progress:
        progress("Semi-global matching (reference)")
    sgbm_disp, sgbm_valid = sgbm_disparity(gray_left, gray_right, window, d_max)
    sgbm_visual = disparity_to_visual(sgbm_disp, sgbm_valid)

    filtered_positions, filtered_colors = build_point_cloud(filtered_disparity, left, filtered_valid)
    sgbm_positions, sgbm_colors = build_point_cloud(sgbm_disp, left, sgbm_valid)

    return {
        "input_left_resized": left,
        "input_right_resized": right,
        "disparity_gray": grayscale,
        "disparity_color": visual,
        "filtered_disparity_visual": filtered_visual,
        "filtered_keep_frac": float(filtered_valid.mean()),
        "sgbm_disparity_visual": sgbm_visual,
        "sgbm_keep_frac": float(sgbm_valid.mean()),
        "point_positions": filtered_positions,
        "point_colors": filtered_colors,
        "sgbm_point_positions": sgbm_positions,
        "sgbm_point_colors": sgbm_colors,
    }
