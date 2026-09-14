"""
Harris corner detection from scratch, plus SSD/NCC patch matching between two
images' corners, and an ORB+BFMatcher baseline for comparison.

Ported from Homework 4 (ECE 661). The Harris detector (Haar-wavelet-like x/y
derivative filters, a second-moment matrix summed over a window, an
image-adaptive k, non-max suppression) is unchanged; the border-unsafe patch
slicing and the double-loop pairwise SSD/NCC computation were replaced with
bounds-checked, vectorized numpy versions, and the original's manually-picked
polygon regions of interest were dropped in favor of matching across the whole
image (or corners are matched by whoever their overall best mutual candidate
is, image-wide). One real bug in the original was also fixed while porting:
it sorted both SSD *and* NCC scores ascending for its greedy assignment, but
NCC is a similarity (higher = better), not a distance -- sorting it ascending
would greedily commit the *worst* matches first. This port sorts by a unified
"lower is better" cost (SSD as-is, negated NCC) instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from scipy.ndimage import maximum_filter

MAX_DIM = 700

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "corners"
SAMPLE_LEFT = SAMPLE_DIR / "left.jpg"
SAMPLE_RIGHT = SAMPLE_DIR / "right.jpg"


def resize_for_demo(img_bgr: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr


def _normalize(image: np.ndarray) -> np.ndarray:
    lo, hi = np.min(image), np.max(image)
    if hi - lo < 1e-9:
        return np.zeros_like(image)
    return (image - lo) / (hi - lo)


def harris_corners(gray: np.ndarray, sigma: float, max_corners: int = 400) -> list[tuple[int, int]]:
    """Returns (x, y) corner locations, strongest-response first."""
    M = int(np.ceil(4 * sigma))
    if M % 2 == 1:
        M += 1
    wave_x = np.hstack([-np.ones((M, M // 2)), np.ones((M, M // 2))])
    wave_y = np.vstack([np.ones((M // 2, M)), -np.ones((M // 2, M))])

    gray_f = gray.astype(np.float64)
    dx = _normalize(cv2.filter2D(gray_f, -1, wave_x))
    dy = _normalize(cv2.filter2D(gray_f, -1, wave_y))

    N = int(np.ceil(5 * sigma))
    neighbor = np.ones((N, N))
    dx2_sum = cv2.filter2D(dx ** 2, -1, neighbor)
    dy2_sum = cv2.filter2D(dy ** 2, -1, neighbor)
    dxdy_sum = cv2.filter2D(dx * dy, -1, neighbor)

    det = dx2_sum * dy2_sum - dxdy_sum ** 2
    tr = dx2_sum + dy2_sum
    k = np.mean(det / (tr + 1e-4) ** 2)
    R = det - k * tr ** 2
    R[R < 0] = 0

    flat_sorted = np.sort(R.flatten())
    idx = max(len(flat_sorted) - 500, 0)
    r_thresh = flat_sorted[idx]

    local_max = maximum_filter(R, size=N) == R
    mask = local_max & (R > r_thresh)
    border = N // 2
    if border > 0:
        mask[:border, :] = False
        mask[-border:, :] = False
        mask[:, :border] = False
        mask[:, -border:] = False

    ys, xs = np.nonzero(mask)
    responses = R[ys, xs]
    order = np.argsort(-responses)[:max_corners]
    return [(int(xs[i]), int(ys[i])) for i in order]


def draw_corners(img_bgr: np.ndarray, corners: list[tuple[int, int]]) -> np.ndarray:
    out = img_bgr.copy()
    for x, y in corners:
        cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
    return out


def _extract_patches(gray: np.ndarray, corners: list[tuple[int, int]], half: int):
    h, w = gray.shape
    patches, kept_idx = [], []
    for i, (x, y) in enumerate(corners):
        if half <= x < w - half and half <= y < h - half:
            patch = gray[y - half:y + half, x - half:x + half].astype(np.float64)
            patches.append(patch.flatten())
            kept_idx.append(i)
    if not patches:
        return np.zeros((0, (2 * half) ** 2)), []
    return np.stack(patches), kept_idx


def _cost_matrix(patches1: np.ndarray, patches2: np.ndarray, dist_type: str) -> np.ndarray:
    if dist_type == "NCC":
        c1 = patches1 - patches1.mean(axis=1, keepdims=True)
        c2 = patches2 - patches2.mean(axis=1, keepdims=True)
        norm1 = np.linalg.norm(c1, axis=1)
        norm2 = np.linalg.norm(c2, axis=1)
        ncc = (c1 @ c2.T) / (norm1[:, None] * norm2[None, :] + 1e-4)
        return -ncc  # lower cost = better match, consistent with SSD below
    sq1 = np.sum(patches1 ** 2, axis=1)[:, None]
    sq2 = np.sum(patches2 ** 2, axis=1)[None, :]
    return sq1 + sq2 - 2 * (patches1 @ patches2.T)


def _greedy_bipartite_match(cost: np.ndarray, max_cost: float | None = None):
    """One-to-one assignment, cheapest pairs first (a symmetric stand-in for the
    original's 'match both directions, keep only mutual agreements')."""
    if cost.size == 0:
        return []
    order = np.argsort(cost.ravel())
    n_cols = cost.shape[1]
    used_rows, used_cols = set(), set()
    matches = []
    for flat in order:
        i, j = divmod(int(flat), n_cols)
        if max_cost is not None and cost[i, j] > max_cost:
            break
        if i in used_rows or j in used_cols:
            continue
        matches.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
    return matches


def match_corners(gray1: np.ndarray, gray2: np.ndarray, corners1, corners2, sigma: float, dist_type: str):
    half = int(np.ceil(5 * sigma))
    patches1, idx1 = _extract_patches(gray1, corners1, half)
    patches2, idx2 = _extract_patches(gray2, corners2, half)
    if len(idx1) == 0 or len(idx2) == 0:
        return []
    cost = _cost_matrix(patches1, patches2, dist_type)
    # NCC below ~0 (unrelated patches) or a very large SSD isn't a real match
    max_cost = -0.5 if dist_type == "NCC" else np.percentile(cost, 5)
    raw_matches = _greedy_bipartite_match(cost, max_cost=max_cost)
    return [(idx1[i], idx2[j]) for i, j in raw_matches]


def draw_matches(img1: np.ndarray, img2: np.ndarray, corners1, corners2, matches) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1 + w2] = img2

    colors = [(0, 0, 225), (225, 0, 0), (0, 200, 0), (0, 165, 255), (200, 0, 200)]
    for n, (i, j) in enumerate(matches):
        p1 = corners1[i]
        p2 = (corners2[j][0] + w1, corners2[j][1])
        cv2.line(combined, p1, p2, colors[n % len(colors)], 1)
    for x, y in corners1:
        cv2.circle(combined, (x, y), 3, (0, 128, 255), -1)
    for x, y in corners2:
        cv2.circle(combined, (x + w1, y), 3, (0, 128, 255), -1)
    return combined


def orb_baseline_matches(gray1: np.ndarray, gray2: np.ndarray, img1: np.ndarray, img2: np.ndarray, max_matches: int = 80) -> np.ndarray:
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return np.hstack([img1, img2])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
    return cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:max_matches], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def run_corner_detection(img_bgr: np.ndarray, sigma: float):
    img_bgr = resize_for_demo(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners = harris_corners(gray, sigma)
    overlay = draw_corners(img_bgr, corners)
    return {"input_resized": img_bgr, "overlay": overlay, "num_corners": len(corners)}


def run_corner_matching(img1_bgr: np.ndarray, img2_bgr: np.ndarray, sigma: float, dist_type: str):
    img1_bgr = resize_for_demo(img1_bgr)
    img2_bgr = resize_for_demo(img2_bgr)
    gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)

    corners1 = harris_corners(gray1, sigma)
    corners2 = harris_corners(gray2, sigma)
    matches = match_corners(gray1, gray2, corners1, corners2, sigma, dist_type)
    combined = draw_matches(img1_bgr, img2_bgr, corners1, corners2, matches)
    orb_combined = orb_baseline_matches(gray1, gray2, img1_bgr, img2_bgr)

    return {
        "corners_overlay_1": draw_corners(img1_bgr, corners1),
        "corners_overlay_2": draw_corners(img2_bgr, corners2),
        "combined": combined,
        "orb_combined": orb_combined,
        "num_corners_1": len(corners1),
        "num_corners_2": len(corners2),
        "num_matches": len(matches),
    }
