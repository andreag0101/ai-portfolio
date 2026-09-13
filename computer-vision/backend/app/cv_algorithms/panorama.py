"""
Panorama stitching: SIFT correspondences -> linear least-squares homography ->
custom RANSAC outlier rejection -> Levenberg-Marquardt nonlinear refinement ->
multi-image compositing with hole-filling interpolation.

Ported from Homework 5 (ECE 661), generalized from a fixed 5-image pipeline to
an arbitrary sequence of N>=2 images anchored at the middle image, and with the
per-pixel Python forward-warp loops replaced by vectorized numpy scatter
operations (the original pixel-by-pixel loop over a full-resolution image would
take tens of seconds per image; the vectorized version is near-instant).
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy import ndimage as nd

MAX_DIM = 900  # resize uploads to keep SIFT + warping responsive


def resize_for_demo(img_bgr: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr


def detect_sift(gray: np.ndarray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des


def draw_keypoints(img_bgr: np.ndarray, kp) -> np.ndarray:
    return cv2.drawKeypoints(img_bgr, kp, None, color=(0, 165, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def sift_matches(kp1, des1, kp2, des2, ratio: float = 0.75, max_matches: int = 300):
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise ValueError("Not enough SIFT keypoints found to match these images.")
    bf = cv2.BFMatcher()
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    good = sorted(good, key=lambda m: m.distance)[:max_matches]
    if len(good) < 8:
        raise ValueError("Not enough confident matches between these two images.")
    coords1 = np.array([kp1[m.queryIdx].pt for m in good])
    coords2 = np.array([kp2[m.trainIdx].pt for m in good])
    return coords1, coords2


def draw_pair_matches(img1: np.ndarray, img2: np.ndarray, coords1, coords2, inliers, outliers) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1 + w2] = img2

    def draw(indices, color):
        for i in indices:
            p1 = (int(coords1[i][0]), int(coords1[i][1]))
            p2 = (int(coords2[i][0]) + w1, int(coords2[i][1]))
            cv2.line(combined, p1, p2, color, 1)
            cv2.circle(combined, p1, 3, color, -1)
            cv2.circle(combined, p2, 3, color, -1)

    draw(outliers, (0, 0, 220))
    draw(inliers, (0, 180, 0))
    return combined


def get_H_LLS(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    n = len(coords1)
    A = np.zeros((2 * n, 8))
    B = np.zeros((2 * n, 1))
    for i in range(n):
        X = np.array([coords1[i][0], coords1[i][1], 1])
        Xp = np.array([coords2[i][0], coords2[i][1], 1])
        A[2 * i, 0:3] = X
        A[2 * i + 1, 3:6] = X
        A[2 * i:2 * i + 2, 6:8] = -np.reshape(Xp[0:2], (2, 1)) @ np.reshape(X[0:2], (1, 2))
        B[2 * i:2 * i + 2] = np.reshape(Xp[0:2].T, (2, 1))
    h, *_ = np.linalg.lstsq(A, B, rcond=None)
    h = np.append(h, 1)
    return h.reshape(3, 3)


def get_inliers(H: np.ndarray, coords1: np.ndarray, coords2: np.ndarray, delta: float):
    ones = np.ones((len(coords1), 1))
    X = np.hstack([coords1, ones])
    Xp_hat = (H @ X.T).T
    Xp_hat = Xp_hat / Xp_hat[:, 2:3]
    dist2 = np.sum((Xp_hat[:, :2] - coords2) ** 2, axis=1)
    inliers = np.where(dist2 < delta ** 2)[0]
    outliers = np.where(dist2 >= delta ** 2)[0]
    return inliers, outliers


def ransac_homography(coords1: np.ndarray, coords2: np.ndarray, const: float = 10, seed: int | None = None):
    rng = np.random.default_rng(seed)
    e, p, n = 0.4, 0.99, 6
    trials = min(int(np.log(1 - p) / np.log(1 - (1 - e) ** n)), 1500)
    sigma = 2
    delta = 3 * sigma * const

    best_inliers, best_outliers, most = np.array([], dtype=int), np.array([], dtype=int), 0
    m = len(coords1)
    for _ in range(trials):
        idx = rng.choice(m, size=n, replace=False)
        H = get_H_LLS(coords1[idx], coords2[idx])
        inliers, outliers = get_inliers(H, coords1, coords2, delta)
        if len(inliers) > most:
            most, best_inliers, best_outliers = len(inliers), inliers, outliers

    if most < 6:
        raise ValueError("RANSAC could not find a consistent homography between these images.")

    H_opt = get_H_LLS(coords1[best_inliers], coords2[best_inliers])
    inliers, outliers = get_inliers(H_opt, coords1, coords2, delta)
    return H_opt, inliers, outliers


def homography_transform(H: np.ndarray, pts_src: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts_src, np.ones((pts_src.shape[0], 1))])
    proj = (H @ pts_h.T).T
    proj /= proj[:, 2:3]
    return proj[:, :2]


def lm_refine(H_init: np.ndarray, coords1: np.ndarray, coords2: np.ndarray, inliers: np.ndarray) -> np.ndarray:
    pts_src, pts_dst = coords1[inliers], coords2[inliers]

    def reprojection_error(H_flat):
        H = H_flat.reshape(3, 3)
        return (homography_transform(H, pts_src) - pts_dst).flatten()

    res = least_squares(reprojection_error, H_init.flatten(), method="lm")
    return res.x.reshape(3, 3)


def apply_homography_forward(H: np.ndarray, img: np.ndarray):
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.ravel(), y.ravel()
    src = np.stack([x, y, np.ones_like(x)])
    dst = H @ src
    dst = dst / dst[2, :]
    return src, dst


def warp_into_canvas(img: np.ndarray, H: np.ndarray, canvas: np.ndarray, offset_x: float, offset_y: float):
    src, dst = apply_homography_forward(H, img)
    dst_x = (dst[0, :] - offset_x).astype(np.int64)
    dst_y = (dst[1, :] - offset_y).astype(np.int64)
    ch, cw = canvas.shape[:2]
    valid = (dst_x >= 0) & (dst_x < cw) & (dst_y >= 0) & (dst_y < ch)
    src_x, src_y = src[0, valid].astype(np.int64), src[1, valid].astype(np.int64)
    canvas[dst_y[valid], dst_x[valid]] = img[src_y, src_x]


def interpolate_holes(sparse: np.ndarray) -> np.ndarray:
    h, w, _ = sparse.shape
    collapse = np.sum(sparse, axis=2)
    mask = np.zeros((h, w), dtype=bool)
    for i, row in enumerate(collapse):
        nz = np.where(row != 0)[0]
        if nz.size > 0:
            mask[i, nz[0]:nz[-1]] = True
    targets = mask & (collapse == 0)
    ind = nd.distance_transform_edt(targets, return_distances=False, return_indices=True)
    return sparse[tuple(ind)]


def warped_corners(H: np.ndarray, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
    proj = H @ corners
    proj /= proj[2, :]
    return proj[:2, :].T


def stitch_panorama(images_bgr: list[np.ndarray], const: float = 10, progress=None) -> dict:
    if len(images_bgr) < 2:
        raise ValueError("Upload at least two overlapping images to stitch a panorama.")

    images_bgr = [resize_for_demo(im) for im in images_bgr]
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images_bgr]
    n = len(images_bgr)

    keypoints, descriptors, keypoint_overlays = [], [], []
    for i in range(n):
        kp, des = detect_sift(grays[i])
        keypoints.append(kp)
        descriptors.append(des)
        keypoint_overlays.append(draw_keypoints(images_bgr[i], kp))

    pairwise_H, match_overlays = [], []
    for i in range(n - 1):
        if progress:
            progress(f"Matching image {i + 1} <-> {i + 2}")
        coords1, coords2 = sift_matches(keypoints[i], descriptors[i], keypoints[i + 1], descriptors[i + 1])
        H_opt, inliers, outliers = ransac_homography(coords1, coords2, const)
        H_refined = lm_refine(H_opt, coords1, coords2, inliers)
        pairwise_H.append(H_refined)
        match_overlays.append(draw_pair_matches(images_bgr[i], images_bgr[i + 1], coords1, coords2, inliers, outliers))

    anchor = n // 2
    H_to_anchor: list[np.ndarray | None] = [None] * n
    H_to_anchor[anchor] = np.eye(3)
    for k in range(anchor - 1, -1, -1):
        H_to_anchor[k] = H_to_anchor[k + 1] @ pairwise_H[k]
    if anchor + 1 < n:
        H_to_anchor[anchor + 1] = np.linalg.inv(pairwise_H[anchor])
    for k in range(anchor + 2, n):
        H_to_anchor[k] = H_to_anchor[k - 1] @ np.linalg.inv(pairwise_H[k - 1])

    all_corners = np.vstack([warped_corners(H_to_anchor[k], images_bgr[k]) for k in range(n)])
    min_x, min_y = all_corners.min(axis=0)
    max_x, max_y = all_corners.max(axis=0)

    canvas_w = int(np.ceil(max_x - min_x)) + 2
    canvas_h = int(np.ceil(max_y - min_y)) + 2
    canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)

    for k in range(n):
        if progress:
            progress(f"Compositing image {k + 1}/{n}")
        warp_into_canvas(images_bgr[k], H_to_anchor[k], canvas, min_x, min_y)

    if progress:
        progress("Filling seams")
    panorama = interpolate_holes(canvas)

    return {
        "keypoint_overlays": keypoint_overlays,
        "match_overlays": match_overlays,
        "panorama": panorama,
    }
