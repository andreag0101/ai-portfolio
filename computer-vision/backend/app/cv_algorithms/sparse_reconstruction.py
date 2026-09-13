"""
Sparse, edge-based 3D reconstruction via two-view epipolar geometry, ported
from Homework 9 (ECE 661) Task 1/2 -- a genuinely different technique from
the dense census/SGM matching in `disparity.py`. Instead of estimating a
disparity for *every* pixel, this: (1) estimates the fundamental matrix from
point correspondences, (2) derives a pair of projective camera matrices from
it, (3) rectifies both images so corresponding points fall on the same row,
(4) finds Canny edge points in the rectified images (interest points
concentrated on object silhouettes, not flat/textured backgrounds), (5)
matches *only those edge points* along their shared row, and (6) triangulates
just the matched edges into 3D. Skipping flat regions entirely -- rather than
matching them and then filtering out bad answers, as the dense approach does
-- is what keeps a noisy background from ever entering the reconstruction.

The original's F-matrix estimation, triangulation, LM refinement, and
epipole-based rectification (`get_H_prime`, `get_H`) are ported faithfully --
the underlying math (normalized 8-point algorithm, canonical camera matrices
from F, linear-least-squares triangulation via SVD) is unchanged. Two things
are different:

1. The original used ~40 manually-clicked correspondences to seed the
   fundamental matrix. There's no equivalent for an arbitrary upload, so this
   uses automatic SIFT matching + RANSAC (scored by Sampson distance against
   the epipolar constraint, the standard error metric for F) instead --
   validated to reach *better* epipolar residuals on the original assignment's
   own photos than the hand-picked points did.
2. The final step of the original's `task1()` has two real bugs: it builds
   `x2_rect` from `edge_coords1_rect` (the same array used for `x1_rect`,
   not `edge_coords2_rect`), and then calls
   `reconstruction_auto(P1, P2_refined, x1_original, x1_original)` -- passing
   the same points as both views. Both together mean the last stage
   triangulates each point against *itself* rather than its match in the
   other image, which would collapse the reconstruction. This port maps each
   image's matched, rectified edge points back to original-image coordinates
   independently (inverting the rectifying homography and undoing the
   canvas-alignment translation) and triangulates the correct pairs.

The canonical camera pair (`get_canonical_cameras`) used for rectification is
a two-view *projective* reconstruction -- there's no known camera calibration,
so it's only defined up to an unknown projective transformation, and its
raw X/Y/Z have essentially arbitrary per-axis scale. The edge points that
actually get displayed, though, are triangulated with the *calibrated*
camera pair instead (`get_calibrated_cameras`, assuming a nominal focal
length): that's Euclidean up to a single unknown *uniform* scale, meaning
relative proportions between axes are meaningful and should be preserved,
not arbitrary. `point_cloud_for_display` rescales with one scale factor
shared across all three axes for exactly this reason -- see its docstring
for a bug an earlier, per-axis version of this had.

Edge-only matching keeps *flat background* out of the reconstruction, but it
does not protect against a second, unrelated failure mode: a camera pair that
barely translated sideways between shots (mostly rotated in place instead).
Diagnosing this on the original assignment's own `Pic_1.jpg`/`Pic_2.jpg` test
photos (which reconstruct as a "starburst" radiating from the camera, however
tightly the edge points are filtered) turned up why per-point filtering can't
fix it: with the two viewing rays to a point nearly parallel, a fraction-of-a-
pixel matching error swings the triangulated depth by an enormous amount --
noise that looks exactly like a well-matched point until it's triangulated.
`get_calibrated_cameras`' essential-matrix decomposition is the part that
actually breaks down here: on that pair it reprojects with tens of px of
median error even on the cleanest SIFT inliers (vs. ~0.1px for the
uncalibrated projective cameras on the same points), regardless of which of
the 4 decomposition candidates or nominal focal length is used -- the
decomposition itself is numerically unstable at near-zero parallax, not
merely "less accurate." `run_sparse_reconstruction` measures the median
triangulation angle across the clean SIFT inlier correspondences (not the
edge points -- see `MIN_RANSAC_INLIERS`'s comment) and raises a clear error
below `MIN_MEDIAN_PARALLAX_DEG` rather than rendering a cloud that looks
broken for reasons that have nothing to do with matching or filtering
quality.

A wrong nominal focal length doesn't just add reprojection error -- it also
distorts the depth axis relative to the lateral (X/Y) axes in the recovered
structure (e.g. a head rendering elongated front-to-back). A per-pair search
for the focal length minimizing reprojection error was tried and reverted:
see `NOMINAL_FOCAL_MULT`'s comment for why it's an ill-posed optimization
here (the bas-relief ambiguity) rather than a reliable fix. Some residual
depth/lateral distortion is an inherent limitation of an assumed, not
measured, focal length, and can't be fully eliminated without real camera
calibration -- though on the deep, stepped architecture of the bundled
sample pair specifically, a meaningful part of the measured depth extent
also reflects genuine scene structure, not distortion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from scipy.optimize import least_squares

from .panorama import (
    resize_for_demo as _resize_for_demo,
    detect_sift,
    sift_matches,
    warp_into_canvas,
    interpolate_holes,
    warped_corners,
)

MAX_DIM = 640

# A validated pair with both plenty of shared texture (for reliable matching)
# and a real sideways baseline (for well-conditioned triangulation) -- not
# every pair of photos of the same scene has both. A narrow baseline
# (camera barely translated, mostly rotated in place) gives an unstable
# reconstruction regardless of matching quality; see the triangulation-angle
# filter in `run_sparse_reconstruction`.
SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "sparse_reconstruction"
SAMPLE_LEFT = SAMPLE_DIR / "left.jpg"
SAMPLE_RIGHT = SAMPLE_DIR / "right.jpg"

# `get_calibrated_cameras` needs a nominal focal length -- there's no real
# calibration for an arbitrary upload. Searching per-pair for whichever focal
# length minimizes 2D reprojection error looks appealing (it can clearly beat
# a fixed guess on that metric) but was tried and reverted: it's an
# ill-posed optimization here, an instance of the classical "bas-relief
# ambiguity" in monocular two-view SfM, where jointly rescaling depth and
# adjusting focal length reproduces nearly the same 2D projections -- so the
# error-minimizing focal length isn't reliably the one that best preserves
# true depth/lateral proportions, and it moved around enough between pairs
# to invalidate the parallax-based quality gate below (a previously
# well-separated validated-good pair measured *lower* parallax than a
# validated-bad one once the focal length was searched per-pair). A single
# fixed value, honestly labeled as a guess, is more predictable: 1.0
# (previously 1.2) was chosen because it measured a much better reprojection
# fit than 1.2 on the validated sample pair (median ~4px vs ~13.5px) without
# introducing this instability.
NOMINAL_FOCAL_MULT = 1.0

# A handful of correspondences can still yield a low Sampson error yet an
# essential-matrix decomposition (see `get_calibrated_cameras`) that doesn't
# actually fit the data -- e.g. 49 RANSAC inliers on a real test pair still
# reprojected with a median error over 10px through the calibrated cameras.
# Below this count, the recovered camera pose is unreliable regardless of how
# clean the matches look.
MIN_RANSAC_INLIERS = 80

# The median angle between the two viewing rays to a triangulated point
# ("parallax"), measured in degrees. Two real test pairs with a camera that
# barely translated (mostly rotated in place between shots) measured
# 0.4-0.8 degrees of median parallax and produced a "starburst" of noise
# radiating from the camera regardless of point filtering afterward -- a
# small pixel matching error swings the triangulated depth wildly when the
# two rays are nearly parallel, which no post-hoc filter can distinguish from
# real depth variation. A validated pair with a real sideways step measured
# ~2.2 degrees under NOMINAL_FOCAL_MULT=1.0 (this was ~4.9 degrees under the
# old 1.2 default -- the parallax measurement is itself sensitive to the
# focal length assumption, so this threshold was recalibrated alongside it,
# not left at its old value). This sits at the midpoint between the two
# clusters, with comfortable margin either direction.
MIN_MEDIAN_PARALLAX_DEG = 1.5


def resize_for_demo(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    return _resize_for_demo(img, max_dim)


# ---------------------------------------------------------------------------
# Fundamental matrix, epipoles, canonical cameras, triangulation
# ---------------------------------------------------------------------------

def normalize_pts(points: np.ndarray):
    """points: (3, N) homogeneous. Returns (normalized_points, T)."""
    mean = np.mean(points, axis=1).reshape(3, 1)
    d_mean = np.mean(np.linalg.norm(points - mean, axis=0))
    s = np.sqrt(2) / (d_mean + 1e-12)
    T = np.array([[s, 0, -s * mean[0, 0]], [0, s, -s * mean[1, 0]], [0, 0, 1]])
    return (points - mean) * s, T


def get_linear_F(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Normalized 8-point algorithm. x1, x2: (3, N) homogeneous, N >= 8."""
    x1_n, T1 = normalize_pts(x1)
    x2_n, T2 = normalize_pts(x2)
    n = x1_n.shape[1]
    A = np.stack([
        x2_n[0] * x1_n[0], x2_n[0] * x1_n[1], x2_n[0],
        x2_n[1] * x1_n[0], x2_n[1] * x1_n[1], x2_n[1],
        x1_n[0], x1_n[1], np.ones(n),
    ], axis=1)
    _, _, vh = np.linalg.svd(A)
    F = vh[-1].reshape(3, 3)
    u, s, vh2 = np.linalg.svd(F)
    s[-1] = 0  # rank-2 condition
    F_conditioned = u @ np.diag(s) @ vh2
    F_denorm = T2.T @ F_conditioned @ T1
    return F_denorm / F_denorm[-1, -1]


def sampson_error(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """First-order approximation of geometric (reprojection) error for the
    epipolar constraint x2^T F x1 = 0 -- the standard error metric for
    scoring fundamental-matrix inliers."""
    Fx1 = F @ x1
    Ftx2 = F.T @ x2
    x2tFx1 = np.sum(x2 * Fx1, axis=0)
    denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
    return x2tFx1 ** 2 / (denom + 1e-12)


def ransac_fundamental_matrix(x1: np.ndarray, x2: np.ndarray, iters: int = 2000,
                               thresh: float = 0.01, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = x1.shape[1]
    best_inliers = np.array([], dtype=int)
    for _ in range(iters):
        idx = rng.choice(n, size=8, replace=False)
        F = get_linear_F(x1[:, idx], x2[:, idx])
        err = sampson_error(F, x1, x2)
        inliers = np.where(err < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    if len(best_inliers) < 8:
        raise ValueError("Not enough consistent matches to estimate the epipolar geometry.")
    F_final = get_linear_F(x1[:, best_inliers], x2[:, best_inliers])
    err = sampson_error(F_final, x1, x2)
    inliers = np.where(err < thresh)[0]
    return F_final, inliers


def get_epipoles(F: np.ndarray):
    u, _, vh = np.linalg.svd(F)
    e1 = vh[-1].reshape(3, 1)
    e2 = u[:, -1].reshape(3, 1)
    return e1 / e1[-1], e2 / e2[-1]


def get_cross_mat(vec: np.ndarray) -> np.ndarray:
    v = vec.ravel()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def get_canonical_cameras(F: np.ndarray, e2: np.ndarray):
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([get_cross_mat(e2) @ F, e2])
    return P1, P2


def get_calibrated_cameras(F: np.ndarray, K: np.ndarray, x1: np.ndarray, x2: np.ndarray):
    """The canonical cameras from F alone (`get_canonical_cameras`) are only
    defined up to an unknown *projective* transform -- correct for triangulating
    the epipolar geometry, but the recovered structure can look arbitrarily
    skewed/flattened rather than like the real object, since nothing pins down
    angles or relative scale between axes. Assuming an approximate focal
    length K (the same kind of "no real calibration, so a nominal one" honest
    assumption already used for the dense disparity point cloud) upgrades F to
    an essential matrix E = K^T F K, whose SVD decomposition gives a camera
    rotation/translation (Euclidean up to one global, unrecoverable scale) --
    standard two-view structure-from-motion. Of the four possible
    decompositions of E, the correct one is the one where triangulated points
    end up in front of *both* cameras (the cheirality check)."""
    E = K.T @ F @ K
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R_a, R_b = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2:3]
    candidates = [(R_a, t), (R_a, -t), (R_b, t), (R_b, -t)]

    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    best_Rt, best_count = candidates[0], -1
    for R, t_cand in candidates:
        P2 = K @ np.hstack([R, t_cand])
        X = img_to_world(x1, x2, P1, P2)
        depth1 = X[2]
        depth2 = (np.hstack([R, t_cand]) @ X)[2]
        count = int(np.sum((depth1 > 0) & (depth2 > 0)))
        if count > best_count:
            best_count, best_Rt = count, (R, t_cand)
    R, t = best_Rt
    P2 = K @ np.hstack([R, t])
    return P1, P2




def img_to_world(x1: np.ndarray, x2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Linear-least-squares triangulation via SVD, batched over all N points
    at once (the original did this with a Python loop calling `np.linalg.svd`
    per point; numpy's SVD batches over leading dimensions, so this is the
    same math, vectorized). x1, x2: (3, N). Returns (4, N) homogeneous."""
    P11, P12, P13 = P1[0], P1[1], P1[2]
    P21, P22, P23 = P2[0], P2[1], P2[2]
    n = x1.shape[1]
    A = np.empty((n, 4, 4))
    A[:, 0, :] = x1[0, :, None] * P13[None, :] - P11[None, :]
    A[:, 1, :] = x1[1, :, None] * P13[None, :] - P12[None, :]
    A[:, 2, :] = x2[0, :, None] * P23[None, :] - P21[None, :]
    A[:, 3, :] = x2[1, :, None] * P23[None, :] - P22[None, :]
    _, _, vh = np.linalg.svd(A)
    X = vh[:, -1, :]
    X = X / X[:, -1:]
    return X.T


def world_to_img(P1: np.ndarray, P2: np.ndarray, X: np.ndarray):
    x1_hat = P1 @ X
    x2_hat = P2 @ X
    return x1_hat / x1_hat[2, :], x2_hat / x2_hat[2, :]


def refine_bundle(x1: np.ndarray, x2: np.ndarray, P1: np.ndarray, P2_init: np.ndarray, X_init: np.ndarray):
    """Levenberg-Marquardt bundle refinement of P2 and the 3D points jointly
    against reprojection error in both views -- a direct port of the
    original's `get_LM_P`."""
    init_params = np.hstack([P2_init.flatten(), X_init[0:3, :].flatten()])

    def cost(params):
        P2 = params[0:12].reshape(3, 4)
        X = params[12:].reshape(3, -1)
        X_h = np.vstack([X, np.ones((1, X.shape[1]))])
        x1_hat, x2_hat = world_to_img(P1, P2, X_h)
        return np.hstack([(x1_hat[:2] - x1[:2]).ravel(), (x2_hat[:2] - x2[:2]).ravel()])

    res = least_squares(cost, init_params, method="lm")
    P2_refined = res.x[0:12].reshape(3, 4)
    X_refined = np.vstack([res.x[12:].reshape(3, -1), np.ones((1, X_init.shape[1]))])
    return P2_refined / P2_refined[-1, -1], X_refined


# ---------------------------------------------------------------------------
# Rectification (epipole-based, Hartley-style)
# ---------------------------------------------------------------------------

def get_H_prime(img_shape, e2: np.ndarray) -> np.ndarray:
    h, w = img_shape[:2]
    x0, y0 = w / 2, h / 2
    e2 = e2.ravel()
    theta = np.arctan(-(e2[1] - y0) / (e2[0] - x0 + 1e-12))
    f = np.abs((e2[0] - x0) * np.cos(theta) - (e2[1] - y0) * np.sin(theta))
    T1 = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    T2 = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]])
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    G = np.array([[1, 0, 0], [0, 1, 0], [-1 / (f + 1e-12), 0, 1]])
    H2 = T2 @ G @ R @ T1
    return H2 / H2[-1, -1]


def get_H_match(P1: np.ndarray, P2: np.ndarray, H2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """H1, chosen (as in the original) so that after rectification,
    corresponding points also line up in the same *column* range as closely
    as possible -- a 1D least-squares fit of the remaining degree of freedom
    in H1 once H2 is fixed."""
    P1_pinv = P1.T @ np.linalg.inv(P1 @ P1.T)
    M = P2 @ P1_pinv
    H0 = H2 @ M
    x1_hat = H0 @ x1
    x1_hat = x1_hat / x1_hat[2, :]
    x2_hat = H2 @ x2
    x2_hat = x2_hat / x2_hat[2, :]
    A = x1_hat.T
    b = x2_hat[0, :].T
    H_a = np.eye(3)
    H_a[0, :] = np.linalg.inv(A.T @ A) @ A.T @ b
    H1 = H_a @ H0
    return H1 / H1[-1, -1]


def rectify_pair(img1: np.ndarray, img2: np.ndarray, H1: np.ndarray, H2: np.ndarray):
    """Forward-warp both images through their rectifying homography into a
    shared canvas (vectorized scatter, reusing the same warp used for
    panorama stitching), and fill small gaps. Returns the two rectified
    images plus the canvas offset (needed later to map matched rectified
    points back to original image coordinates)."""
    corners1 = warped_corners(H1, img1)
    corners2 = warped_corners(H2, img2)
    all_corners = np.vstack([corners1, corners2])
    min_x, min_y = all_corners.min(axis=0)
    max_x, max_y = all_corners.max(axis=0)
    w = int(np.ceil(max_x - min_x)) + 2
    h = int(np.ceil(max_y - min_y)) + 2
    # cap the rectified canvas size -- an extreme epipole geometry can blow
    # this up arbitrarily large
    scale = min(1.0, 1600 / max(w, h))
    canvas1 = np.zeros((h, w, 3), np.uint8)
    canvas2 = np.zeros((h, w, 3), np.uint8)
    warp_into_canvas(img1, H1, canvas1, min_x, min_y)
    warp_into_canvas(img2, H2, canvas2, min_x, min_y)
    rect1 = interpolate_holes(canvas1)
    rect2 = interpolate_holes(canvas2)
    if scale < 1.0:
        rect1 = cv2.resize(rect1, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rect2 = cv2.resize(rect2, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rect1, rect2, float(min_x), float(min_y), scale


# ---------------------------------------------------------------------------
# Edge detection + row-constrained matching
# ---------------------------------------------------------------------------

def canny_interest_points(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    return cv2.Canny(blurred, 100, 300)


def _extract_patches(gray: np.ndarray, ys: np.ndarray, xs: np.ndarray, half: int) -> np.ndarray:
    """Vectorized window extraction: (K, 2half+1, 2half+1) for K centers."""
    offsets = np.arange(-half, half + 1)
    dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
    rows = ys[:, None, None] + dy[None]
    cols = xs[:, None, None] + dx[None]
    return gray[rows, cols].astype(np.float64)


def match_edges_along_rows(gray1: np.ndarray, gray2: np.ndarray, edges1: np.ndarray, edges2: np.ndarray,
                            window: int = 11, row_tol: int = 3, max_points: int = 1200, seed: int = 0):
    """For each edge pixel in image 1, search edge pixels in image 2 within
    +/- row_tol rows (rectified images: true matches sit on the same row, a
    tolerance absorbs small rectification error) and keep the best SSD match.
    A direct port of the original's `distance_metric`, vectorized: the
    original's innermost per-candidate loop is replaced by extracting every
    candidate's window at once and scoring them with one broadcasted
    subtraction instead of a Python loop over candidates."""
    half = window // 2
    h, w = gray1.shape

    ys1, xs1 = np.nonzero(edges1)
    ys2, xs2 = np.nonzero(edges2)
    # keep points with a full window in bounds
    keep1 = (xs1 >= half) & (xs1 < w - half) & (ys1 >= half) & (ys1 < h - half)
    keep2 = (xs2 >= half) & (xs2 < w - half) & (ys2 >= half) & (ys2 < h - half)
    ys1, xs1 = ys1[keep1], xs1[keep1]
    ys2, xs2 = ys2[keep2], xs2[keep2]

    rng = np.random.default_rng(seed)
    if len(xs1) > max_points:
        sel = rng.choice(len(xs1), size=max_points, replace=False)
        ys1, xs1 = ys1[sel], xs1[sel]
    if len(xs2) == 0 or len(xs1) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    order2 = np.argsort(ys2)
    ys2_sorted, xs2_sorted = ys2[order2], xs2[order2]

    matched1, matched2 = [], []
    for y1, x1 in zip(ys1, xs1):
        lo = np.searchsorted(ys2_sorted, y1 - row_tol, side="left")
        hi = np.searchsorted(ys2_sorted, y1 + row_tol, side="right")
        if lo >= hi:
            continue
        cand_y, cand_x = ys2_sorted[lo:hi], xs2_sorted[lo:hi]
        patch1 = gray1[y1 - half:y1 + half + 1, x1 - half:x1 + half + 1].astype(np.float64)
        patches2 = _extract_patches(gray2, cand_y, cand_x, half)
        ssd = np.sum((patches2 - patch1[None]) ** 2, axis=(1, 2))
        best = np.argmin(ssd)
        matched1.append((x1, y1))
        matched2.append((cand_x[best], cand_y[best]))

    return np.array(matched1, dtype=np.float64), np.array(matched2, dtype=np.float64)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def draw_keypoint_matches(img1, img2, coords1, coords2, inliers) -> np.ndarray:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h = max(h1, h2)
    combined = np.zeros((h, w1 + w2, 3), np.uint8)
    combined[:h1, :w1] = img1
    combined[:h2, w1:w1 + w2] = img2
    for i in inliers:
        p1 = (int(coords1[i][0]), int(coords1[i][1]))
        p2 = (int(coords2[i][0]) + w1, int(coords2[i][1]))
        cv2.line(combined, p1, p2, (0, 180, 0), 1)
    return combined


def draw_edge_overlay(img_bgr: np.ndarray, edges: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    out[edges > 0] = (0, 0, 255)
    return out


def draw_edge_matches(rect1, rect2, m1, m2, stride: int = 15) -> np.ndarray:
    h1, w1 = rect1.shape[:2]
    combined = np.hstack([rect1, rect2]).copy()
    rng = np.random.default_rng(0)
    for i in range(0, len(m1), stride):
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        p1 = (int(m1[i, 0]), int(m1[i, 1]))
        p2 = (int(m2[i, 0]) + w1, int(m2[i, 1]))
        cv2.circle(combined, p1, 2, color, -1)
        cv2.circle(combined, p2, 2, color, -1)
        cv2.line(combined, p1, p2, color, 1)
    return combined


def point_cloud_for_display(X: np.ndarray, colors: np.ndarray):
    """X: (3, N) triangulated with the *calibrated* camera pair
    (`get_calibrated_cameras`), so this structure is Euclidean up to a single
    unknown uniform scale -- not an unknown per-axis scale. Purely for the 3D
    viewer: drop the most extreme points and rescale to a comparable visual
    range, using *one* scale factor shared by all three axes.

    An earlier version scaled each axis independently to the same nominal
    range (each axis's own 95th-percentile spread mapped to 300 units). That
    is only valid for the uncalibrated, purely *projective* reconstruction
    (11-DOF camera, arbitrary per-axis scale) -- for this calibrated
    reconstruction it silently distorts the recovered shape's proportions
    whenever the true depth extent differs from the lateral extent (e.g. a
    head rendering elongated front-to-back), which per-axis normalization
    will do to *any* input, correct or not, since it forcibly stretches
    whichever axis happens to have the smaller true spread.

    A handful of triangulated points end up with a homogeneous coordinate
    very close to zero (a near-degenerate triangulation, usually from a
    mismatched edge pair rather than a real "far away" point) and blow up to
    huge values after the projective normalize-by-last-coordinate step. Those
    have to be dropped, not just rescaled: rescaling by a percentile-based
    factor still leaves them enormously far outside the cluster of good
    points, which then dominates the camera's auto-fit view and hides the
    actual structure -- the same "background overwhelms it" symptom this
    whole sparse approach was meant to avoid, just relocated to a few
    outliers instead of the whole background.
    """
    if X.shape[1] == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    median = np.median(X, axis=1, keepdims=True)
    dist = np.linalg.norm(X - median, axis=0)
    keep = dist < np.percentile(dist, 90)
    X, colors = X[:, keep], colors[keep]

    centered = X - np.median(X, axis=1, keepdims=True)
    scale = np.percentile(np.abs(centered), 95)  # one scalar for all 3 axes -- preserves true proportions
    scale = scale if scale > 1e-9 else 1.0
    normalized = (centered / scale) * 300
    positions = normalized.T.astype(np.float32)
    positions[:, 2] *= -1  # so "closer to camera 1" reads as toward the viewer
    return positions, colors.astype(np.uint8)


def run_sparse_reconstruction(img1_bgr: np.ndarray, img2_bgr: np.ndarray, progress=None):
    img1 = resize_for_demo(img1_bgr)
    img2 = resize_for_demo(img2_bgr)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if progress:
        progress("Finding correspondences (SIFT)")
    kp1, des1 = detect_sift(gray1)
    kp2, des2 = detect_sift(gray2)
    coords1, coords2 = sift_matches(kp1, des1, kp2, des2, max_matches=2000)
    x1_all = np.vstack([coords1.T, np.ones(len(coords1))])
    x2_all = np.vstack([coords2.T, np.ones(len(coords2))])

    if progress:
        progress("Estimating the fundamental matrix (RANSAC)")
    F, inlier_idx = ransac_fundamental_matrix(x1_all, x2_all)
    if len(inlier_idx) < MIN_RANSAC_INLIERS:
        raise ValueError(
            f"Only {len(inlier_idx)} reliable correspondences found between these photos -- too few to "
            "estimate the epipolar geometry (and the camera pose derived from it) robustly. Try a pair "
            "with more shared texture/detail, or a smaller viewpoint change between shots."
        )
    keypoint_match_overlay = draw_keypoint_matches(img1, img2, coords1, coords2, inlier_idx)

    x1_in = x1_all[:, inlier_idx]
    x2_in = x2_all[:, inlier_idx]
    _, e2 = get_epipoles(F)
    P1, P2 = get_canonical_cameras(F, e2)
    X_init = img_to_world(x1_in, x2_in, P1, P2)

    rng = np.random.default_rng(0)
    lm_n = min(120, x1_in.shape[1])
    lm_sel = rng.choice(x1_in.shape[1], size=lm_n, replace=False)

    # Recover a calibrated (nominal-focal-length) camera pair now, on the
    # clean SIFT inlier correspondences, and check the parallax they
    # triangulate with -- before spending time on rectification and edge
    # matching. This has to use the *inlier keypoints*, not the edge points
    # matched later: edge matches include enough SSD/occlusion mismatches
    # (some even landing behind one of the cameras) that their triangulation
    # angle distribution is dominated by matching noise rather than the
    # actual camera geometry, which made an earlier version of this check
    # wildly unreliable (e.g. it measured >90 degrees of "parallax" on a
    # pair that a clean-point diagnostic put at 0.6 degrees).
    h1, w1 = img1.shape[:2]
    K = np.array([[NOMINAL_FOCAL_MULT * w1, 0, w1 / 2], [0, NOMINAL_FOCAL_MULT * w1, h1 / 2], [0, 0, 1]])
    P1_cal, P2_cal = get_calibrated_cameras(F, K, x1_in[:, lm_sel], x2_in[:, lm_sel])
    X_cal = img_to_world(x1_in[:, lm_sel], x2_in[:, lm_sel], P1_cal, P2_cal)
    R_cal = np.linalg.inv(K) @ P2_cal[:, :3]
    t_cal = np.linalg.inv(K) @ P2_cal[:, 3:4]
    cam2_center = (-R_cal.T @ t_cal).ravel()
    ray1 = X_cal[:3]  # camera 1 is at the world origin
    ray2 = X_cal[:3] - cam2_center[:, None]
    cos_angle = np.sum(ray1 * ray2, axis=0) / (np.linalg.norm(ray1, axis=0) * np.linalg.norm(ray2, axis=0) + 1e-12)
    median_parallax = float(np.median(np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))))
    if median_parallax < MIN_MEDIAN_PARALLAX_DEG:
        raise ValueError(
            f"This pair's viewing angle barely changes between shots (median parallax {median_parallax:.1f}°, "
            f"reconstructible pairs need roughly {MIN_MEDIAN_PARALLAX_DEG:.0f}°+) -- the camera moved mostly by "
            "rotating in place rather than shifting sideways. Triangulation is fundamentally unstable in this "
            "regime: even pixel-perfect matches swing wildly in depth, which no amount of point filtering can "
            "fix after the fact. Retake the pair with a clear sideways step between the two shots (a few inches "
            "to a foot, not just a tilt/pan), or use the sample pair."
        )

    if progress:
        progress("Refining camera geometry (Levenberg-Marquardt)")
    P2_refined, _ = refine_bundle(x1_in[:, lm_sel], x2_in[:, lm_sel], P1, P2, X_init[:, lm_sel])
    e2_refined = P2_refined[:, -1:]

    if progress:
        progress("Rectifying the image pair")
    H2 = get_H_prime(img2.shape, e2_refined)
    H1 = get_H_match(P1, P2_refined, H2, x1_in[:, lm_sel], x2_in[:, lm_sel])
    rect1, rect2, min_x, min_y, rect_scale = rectify_pair(img1, img2, H1, H2)

    if progress:
        progress("Detecting edges in the rectified pair")
    edges1 = canny_interest_points(rect1)
    edges2 = canny_interest_points(rect2)

    if progress:
        progress("Matching edge points row-by-row")
    rect_gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    rect_gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)
    m1_rect, m2_rect = match_edges_along_rows(rect_gray1, rect_gray2, edges1, edges2)
    edge_match_overlay = draw_edge_matches(rect1, rect2, m1_rect, m2_rect) if len(m1_rect) else np.hstack([rect1, rect2])

    if len(m1_rect) < 8:
        raise ValueError("Not enough matched edge points to reconstruct -- try a pair with a clearer, more textured object.")

    if progress:
        progress("Triangulating matched edges into 3D")
    # map matched rectified points back to original-image coordinates: undo
    # the canvas-alignment translation, then invert the rectifying homography
    H1_inv, H2_inv = np.linalg.inv(H1), np.linalg.inv(H2)
    m1_canvas = np.vstack([m1_rect.T / rect_scale + [[min_x], [min_y]], np.ones(len(m1_rect))])
    m2_canvas = np.vstack([m2_rect.T / rect_scale + [[min_x], [min_y]], np.ones(len(m2_rect))])
    x1_edges = H1_inv @ m1_canvas
    x1_edges /= x1_edges[2, :]
    x2_edges = H2_inv @ m2_canvas
    x2_edges /= x2_edges[2, :]

    # P1_cal, P2_cal (the calibrated camera pair) and cam2_center were already
    # recovered above, before rectification -- reused here rather than
    # recomputed. The canonical (projective-only) P1, P2 used for
    # rectification would triangulate an arbitrarily skewed/flattened shape;
    # see `get_calibrated_cameras`.
    X_edges = img_to_world(x1_edges, x2_edges, P1_cal, P2_cal)

    # sanity filter: reject the worst-reprojecting matches (mismatches,
    # occlusions) rather than trusting every triangulated edge equally
    x1_hat, x2_hat = world_to_img(P1_cal, P2_cal, X_edges)
    err = np.linalg.norm(x1_hat[:2] - x1_edges[:2], axis=0) + np.linalg.norm(x2_hat[:2] - x2_edges[:2], axis=0)
    good = err < np.percentile(err, 90)

    # a *small* triangulation angle for an individual point (the two viewing
    # rays nearly parallel) is a second, independent failure mode that
    # reprojection error alone won't catch -- a fraction-of-a-pixel matching
    # error still reprojects almost perfectly, yet swings that point's
    # triangulated depth wildly along the ray. This is a per-point cleanup
    # pass; the pair-level version of the same check already ran above
    # (`median_parallax`) using the clean SIFT inliers, before any of this
    # edge-matching work was done.
    ray1 = X_edges[:3]  # camera 1 is at the world origin
    ray2 = X_edges[:3] - cam2_center[:, None]
    cos_angle = np.sum(ray1 * ray2, axis=0) / (np.linalg.norm(ray1, axis=0) * np.linalg.norm(ray2, axis=0) + 1e-12)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    good = good & (angle_deg > 1.0)

    h1, w1 = img1.shape[:2]
    px = np.clip(x1_edges[0, good].astype(np.int32), 0, w1 - 1)
    py = np.clip(x1_edges[1, good].astype(np.int32), 0, h1 - 1)
    colors_bgr = img1[py, px]
    colors_rgb = colors_bgr[:, ::-1]

    positions, colors = point_cloud_for_display(X_edges[:3, good], colors_rgb)

    return {
        "left": img1,
        "right": img2,
        "num_sift_matches": int(x1_all.shape[1]),
        "num_ransac_inliers": int(len(inlier_idx)),
        "keypoint_match_overlay": keypoint_match_overlay,
        "rectified_left": rect1,
        "rectified_right": rect2,
        "edges_left": cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR),
        "edges_right": cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR),
        "edge_match_overlay": edge_match_overlay,
        "num_edge_matches": int(len(m1_rect)),
        "num_reconstructed": int(good.sum()),
        "median_parallax_deg": median_parallax,
        "point_positions": positions,
        "point_colors": colors,
    }
