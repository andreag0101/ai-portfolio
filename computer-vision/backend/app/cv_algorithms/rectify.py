"""
Planar homography: automatic quadrilateral detection + 4-point DLT homography,
used two ways -- "straighten" (Homework 3: remove projective distortion from a
photographed plane) and "insert" (Homework 2: warp an image into a planar
region of another photo, e.g. a picture frame or blank screen).

The original assignments located the plane's four corners by having the user
click them by hand (see the sibling `get_coords.py` scripts in Homework 2/3).
This port replaces that manual step with automatic quadrilateral detection
(Canny edges -> contours -> polygon approximation, the same technique document
scanners use) so the whole pipeline runs on an upload with no clicking. The
homography math itself (`get_H`, a direct linear transform from four point
correspondences) and the forward-warp-with-scatter compositing are the same
technique as the original `main.py` files, vectorized for speed.

Candidate quads are filtered by "rectangularity" (contour area vs. the area of
their own minimum-area bounding rectangle) and interior-angle sanity, which
rejects contours broken by occlusion (e.g. a hand gripping the object) that
would otherwise win on raw area alone. Photos with nested rectangles (a picture
frame's outer edge, its mat, and the photo inside it are all valid quads) are
inherently ambiguous about which one the user means, so `detect_quads` returns
several ranked candidates rather than committing to a single guess -- the API
exposes these via `/api/rectify/candidates` and the frontend lets the user
pick one instead of always trusting the top-ranked guess.
"""
from __future__ import annotations

import numpy as np
import cv2

from .panorama import get_H_LLS as get_H, interpolate_holes

MAX_DIM = 1000


def resize_for_demo(img_bgr: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 arbitrary points into [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _interior_angles(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    angles = np.zeros(n)
    for i in range(n):
        prev_v = pts[i - 1] - pts[i]
        next_v = pts[(i + 1) % n] - pts[i]
        cos_a = np.dot(prev_v, next_v) / (np.linalg.norm(prev_v) * np.linalg.norm(next_v) + 1e-9)
        angles[i] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    return angles


def _rectangularity(pts: np.ndarray) -> float:
    """1.0 for a perfect rectangle; drops sharply for occluded/skewed quadrilaterals
    (contour area vs. the area of its own minimum-area bounding rectangle)."""
    rect = cv2.minAreaRect(pts)
    rect_area = rect[1][0] * rect[1][1]
    if rect_area <= 0:
        return 0.0
    return float(cv2.contourArea(pts) / rect_area)


MIN_RECTANGULARITY = 0.75
MIN_ANGLE, MAX_ANGLE = 35, 145


def _is_clean_quad(pts: np.ndarray) -> bool:
    angles = _interior_angles(pts)
    if angles.min() < MIN_ANGLE or angles.max() > MAX_ANGLE:
        return False
    return _rectangularity(pts) >= MIN_RECTANGULARITY


def _candidate_quads(edges: np.ndarray, img_area: float):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.03 * img_area or area > 0.98 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        for eps_frac in (0.01, 0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(c, eps_frac * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                pts = approx.reshape(4, 2).astype(np.float32)
                if _is_clean_quad(pts):
                    candidates.append((area, pts))
                break
    return candidates


def _dedupe_quads(candidates, dist_thresh: float = 15.0):
    """Collapse near-identical quads found across different Canny thresholds,
    keeping the largest (most complete) version of each."""
    candidates = sorted(candidates, key=lambda t: -t[0])
    kept: list[tuple[float, np.ndarray]] = []
    for area, pts in candidates:
        ordered = order_corners(pts)
        is_dup = False
        for _, kept_pts in kept:
            if np.mean(np.linalg.norm(ordered - kept_pts, axis=1)) < dist_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append((area, ordered))
    return kept


def detect_quads(img_bgr: np.ndarray, k: int = 5) -> list[np.ndarray]:
    """Find up to k distinct, plausibly-rectangular quadrilaterals, largest first."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    img_area = float(gray.shape[0] * gray.shape[1])

    all_candidates = []
    for low, high in ((50, 150), (30, 100), (75, 200), (20, 60)):
        edges = cv2.Canny(blur, low, high)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        all_candidates.extend(_candidate_quads(edges, img_area))

    deduped = _dedupe_quads(all_candidates)
    if deduped:
        return [pts for _, pts in deduped[:k]]

    # fallback: largest external contour's minimum-area rotated rectangle,
    # even if it doesn't pass the rectangularity/angle checks
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if 0.03 * img_area < cv2.contourArea(c) < 0.98 * img_area]
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return [order_corners(box)]


def detect_quad(img_bgr: np.ndarray) -> np.ndarray | None:
    """Find the single best-guess quadrilateral in the image, or None."""
    quads = detect_quads(img_bgr, k=1)
    return quads[0] if quads else None


def draw_quad_overlay(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    pts = quad.astype(np.int32)
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=max(2, img_bgr.shape[1] // 300))
    for p in pts:
        cv2.circle(overlay, tuple(p), max(4, img_bgr.shape[1] // 200), (0, 140, 255), -1)
    return overlay


def rect_dimensions(quad: np.ndarray) -> tuple[int, int]:
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    return max(width, 2), max(height, 2)


def _forward_warp_scatter(src_img: np.ndarray, H: np.ndarray, canvas: np.ndarray):
    """Map every source pixel through H and scatter it into canvas (in place)."""
    h, w = src_img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.ravel(), y.ravel()
    src_hc = np.stack([x, y, np.ones_like(x)])
    dst_hc = H @ src_hc
    dst_hc = dst_hc / dst_hc[2, :]
    dst_x = np.round(dst_hc[0, :]).astype(np.int64)
    dst_y = np.round(dst_hc[1, :]).astype(np.int64)
    ch, cw = canvas.shape[:2]
    valid = (dst_x >= 0) & (dst_x < cw) & (dst_y >= 0) & (dst_y < ch)
    canvas[dst_y[valid], dst_x[valid]] = src_img[y[valid], x[valid]]


def dewarp_quad(img_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Straighten the plane bounded by `quad` into a fronto-parallel rectangle."""
    width, height = rect_dimensions(quad)
    rect_corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # H maps the ideal rectangle -> the distorted quad (matches the original
    # Homework 3 convention); Hinv then maps distorted image pixels forward
    # into the straightened output, matching the original's scatter warp.
    H = get_H(rect_corners, quad)
    H_inv = np.linalg.inv(H)

    canvas = np.zeros((height, width, 3), np.uint8)
    _forward_warp_scatter(img_bgr, H_inv, canvas)
    return interpolate_holes(canvas)


def insert_into_quad(dest_bgr: np.ndarray, quad: np.ndarray, source_bgr: np.ndarray) -> np.ndarray:
    """Warp `source_bgr` into the quadrilateral region of `dest_bgr`."""
    h, w = source_bgr.shape[:2]
    src_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    H = get_H(src_corners, quad)

    result = dest_bgr.copy()
    mask = np.zeros(dest_bgr.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)

    warped = np.zeros_like(dest_bgr)
    _forward_warp_scatter(source_bgr, H, warped)

    warped_filled = interpolate_holes(warped)
    region = mask.astype(bool)
    result[region] = warped_filled[region]
    return result


NOT_FOUND_MSG = (
    "Couldn't find a clear quadrilateral in this photo. Try a photo where the plane's "
    "edges contrast clearly with the background, and aren't blocked by hands, cables, etc."
)


def find_candidates(img_bgr: np.ndarray, k: int = 5):
    """Resize + detect; return the resized image and up to k ranked candidate quads."""
    img_bgr = resize_for_demo(img_bgr)
    quads = detect_quads(img_bgr, k=k)
    return img_bgr, quads


def run_dewarp(img_bgr: np.ndarray, quad: np.ndarray | None = None):
    img_bgr = resize_for_demo(img_bgr)
    if quad is None:
        quad = detect_quad(img_bgr)
        if quad is None:
            raise ValueError(NOT_FOUND_MSG)
    else:
        quad = order_corners(np.asarray(quad, dtype=np.float32))
    rectified = dewarp_quad(img_bgr, quad)
    overlay = draw_quad_overlay(img_bgr, quad)
    return {"input_resized": img_bgr, "detected_quad_overlay": overlay, "rectified": rectified}


def run_insert(dest_bgr: np.ndarray, source_bgr: np.ndarray, quad: np.ndarray | None = None):
    dest_bgr = resize_for_demo(dest_bgr)
    if quad is None:
        quad = detect_quad(dest_bgr)
        if quad is None:
            raise ValueError(NOT_FOUND_MSG)
    else:
        quad = order_corners(np.asarray(quad, dtype=np.float32))
    composited = insert_into_quad(dest_bgr, quad, source_bgr)
    overlay = draw_quad_overlay(dest_bgr, quad)
    return {"dest_resized": dest_bgr, "detected_quad_overlay": overlay, "composited": composited}
