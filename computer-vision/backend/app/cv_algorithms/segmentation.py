"""
Image segmentation: iterative Otsu thresholding (per-channel RGB or local-variance
texture) followed by morphological cleanup and contour extraction.

Ported from Homework 6 (ECE 661). The algorithm (iterative Otsu on foreground,
structuring-element morphology, boundary-pixel contours) is unchanged from the
original; the per-pixel Python loops have been replaced with equivalent
vectorized numpy/scipy operations so a full-resolution upload processes in a
fraction of a second instead of minutes.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter


def otsu_threshold_mask(channel: np.ndarray, bins: int, flip: bool) -> np.ndarray:
    """Single-pass Otsu: pick the threshold maximizing between-class variance,
    return a binary foreground mask."""
    counts, edges = np.histogram(channel, bins=bins)
    N = counts.sum()
    if N == 0:
        return np.zeros_like(channel, dtype=bool)
    p = counts / N
    levels = np.arange(bins)

    omega_0 = np.cumsum(p)
    omega_1 = 1.0 - omega_0
    cum_mean = np.cumsum(levels * p)
    total_mean = cum_mean[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mu_0 = np.where(omega_0 > 0, cum_mean / omega_0, 0)
        mu_1 = np.where(omega_1 > 0, (total_mean - cum_mean) / omega_1, 0)
        sigma_b2 = omega_0 * omega_1 * (mu_1 - mu_0) ** 2

    sigma_b2 = sigma_b2[:-2]  # matches original range(L-2)
    if sigma_b2.size == 0:
        k_star = edges[len(edges) // 2]
    else:
        k_star = edges[int(np.argmax(sigma_b2)) + 1]

    return (channel < k_star) if flip else (channel > k_star)


def otsu_iterative_mask(channel: np.ndarray, bins: int, flip: bool, iterations: int,
                         init_mask: np.ndarray) -> np.ndarray:
    """Repeatedly re-run Otsu on the current foreground to refine the mask."""
    current = init_mask
    for _ in range(iterations):
        foreground = channel * current
        new_mask = otsu_threshold_mask(foreground, bins, flip)
        current = np.logical_and(current, new_mask)
    return current


def segment_rgb(img_bgr: np.ndarray, bins: int, flip: bool, iterations) -> dict[str, np.ndarray]:
    """Per-channel iterative Otsu, combined with logical AND across channels."""
    if isinstance(iterations, int):
        iterations = [iterations, iterations, iterations]
    B, G, R = cv2.split(img_bgr)
    channels = {"B": B, "G": G, "R": R}

    init_masks = {c: otsu_threshold_mask(v, bins, flip) for c, v in channels.items()}
    init_combined = np.logical_and.reduce(list(init_masks.values()))

    final_masks = {}
    for (c, v), it in zip(channels.items(), iterations):
        final_masks[c] = otsu_iterative_mask(v, bins, flip, it, init_combined)
    combined = np.logical_and.reduce(list(final_masks.values()))

    return {
        "B": final_masks["B"], "G": final_masks["G"], "R": final_masks["R"],
        "combined": combined, "combined_init": init_combined,
    }


def _local_variance(gray: np.ndarray, window: int) -> np.ndarray:
    mean = uniform_filter(gray.astype(np.float64), size=window, mode="constant", cval=0.0)
    mean_sq = uniform_filter((gray.astype(np.float64)) ** 2, size=window, mode="constant", cval=0.0)
    return mean_sq - mean ** 2


def segment_texture(img_bgr: np.ndarray, flip: bool, iterations) -> dict[str, np.ndarray]:
    """Local-variance texture map at 3 window sizes, each Otsu-segmented, combined."""
    if isinstance(iterations, int):
        iterations = [iterations, iterations, iterations]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    windows = [3, 5, 7]
    masks, init_masks = {}, {}
    for w, it in zip(windows, iterations):
        variance = _local_variance(gray, w)
        init_mask = otsu_threshold_mask(variance, 256, flip)
        masks[w] = otsu_iterative_mask(variance, 256, flip, it, init_mask)
        init_masks[w] = init_mask

    combined = np.logical_and(masks[3], np.logical_and(masks[5], masks[7]))
    combined_init = np.logical_and(init_masks[3], np.logical_and(init_masks[5], init_masks[7]))
    return {"w3": masks[3], "w5": masks[5], "w7": masks[7],
            "combined": combined, "combined_init": combined_init}


def dilation(mask: np.ndarray, n: int) -> np.ndarray:
    return maximum_filter(mask.astype(np.uint8), size=n, mode="constant", cval=0) > 0


def erosion(mask: np.ndarray, n: int) -> np.ndarray:
    return minimum_filter(mask.astype(np.uint8), size=n, mode="constant", cval=0) == 1


def morphological(mask: np.ndarray, operation: str, n: int = 3) -> np.ndarray:
    if operation == "opening":
        return dilation(erosion(mask, n), n)
    return erosion(dilation(mask, n), n)


def extract_contour(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels: where the mask value differs from its right/bottom neighbor."""
    m = mask.astype(bool)
    contour = np.zeros(m.shape, dtype=np.uint8)
    diff_down = m[:-1, :] != m[1:, :]
    diff_right = m[:, :-1] != m[:, 1:]
    body = np.zeros_like(m)
    body[:-1, :-1] = diff_down[:, :-1] | diff_right[:-1, :]
    contour[body] = 255
    return contour


def mask_to_u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8)) * 255


MAX_DIM = 700  # keep interactive demo responsive on large uploads


def resize_for_demo(img_bgr: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr


def run_segmentation(img_bgr: np.ndarray, mode: str, bins: int, flip: bool, iterations: int) -> dict[str, np.ndarray]:
    """Top-level entry point used by the API route. mode is 'rgb' or 'texture'."""
    img_bgr = resize_for_demo(img_bgr)
    if mode == "texture":
        masks = segment_texture(img_bgr, flip, iterations)
    else:
        masks = segment_rgb(img_bgr, bins, flip, iterations)

    combined = masks["combined"]
    none_contour = extract_contour(combined)
    opening_mask = morphological(combined, "opening")
    opening_contour = extract_contour(opening_mask)
    closing_mask = morphological(combined, "closing")
    closing_contour = extract_contour(closing_mask)

    return {
        "input_resized": img_bgr,
        "mask": mask_to_u8(combined),
        "mask_before_iteration": mask_to_u8(masks["combined_init"]),
        "contour": none_contour,
        "opening_mask": mask_to_u8(opening_mask),
        "opening_contour": opening_contour,
        "closing_mask": mask_to_u8(closing_mask),
        "closing_contour": closing_contour,
    }
