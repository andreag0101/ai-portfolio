"""
Camera calibration via Zhang's method, ported from Homework 8 (ECE 661).

The calibration pattern is a bespoke 4x5 grid of isolated black squares, not
a standard chessboard tessellation, so OpenCV's findChessboardCorners can't
be used. The original's own from-scratch corner detector is what's ported
here: Canny edges -> Hough line detection -> classify lines as near-vertical
or near-horizontal -> every vertical/horizontal line intersection is a corner
candidate -> k-means clusters those candidates down to the 80 true corners
(20 squares x 4 corners each). That detector is unchanged except for
vectorizing one per-pixel threshold loop. Everything downstream -- per-image
homography (DLT), the absolute conic from multiple homographies (Zhang's
closed-form), intrinsics/extrinsics extraction, and Levenberg-Marquardt
refinement -- is a direct port; none of that had per-pixel loops to begin with.

One real bug was fixed while porting: the closed-form intrinsics extraction
(`_omega_to_K`) computed one denominator as `omega[0,1]*omega[0,1]**2`
(i.e. omega[0,1]**3) where Zhang's formula -- and the very next line of the
*same original function* -- uses `omega[0,0]*omega[1,1] - omega[0,1]**2`.
That's an internal inconsistency, not a deliberate variant. Since only LM
refinement's *initial guess* depends on this closed-form step, the original's
bug was likely masked by LM converging to a good answer anyway from a
slightly-off start; this port uses the corrected formula so the closed-form
estimate (also shown on its own, before refinement) is actually correct.

Because the pattern is bespoke, calibrating requires 3+ photos of *this*
specific printed pattern (served by the API for printing), or the bundled
sample dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io

from .panorama import get_H_LLS

CALIB_DIR = Path(__file__).resolve().parents[2] / "data" / "calibration"
PATTERN_PATH = CALIB_DIR / "CalibrationPattern.jpg"
SAMPLE_DIR = CALIB_DIR / "Dataset1"

GRID_COLS, GRID_ROWS = 8, 10  # 4x5 squares, 2 corners per side per square
N_CORNERS = GRID_COLS * GRID_ROWS
RECT_W, RECT_H = 350, 450  # canonical sorting rectangle, arbitrary units

MAX_DIM = 900


def resize_for_demo(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def detect_corners(img_bgr: np.ndarray) -> np.ndarray:
    """Hough line intersections + k-means -> the 80 corner candidates, unordered."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).copy()
    gray[gray > 50] = 255
    edges = cv2.Canny(gray, 100, 200)
    hough_lines = cv2.HoughLines(edges, 1, np.pi / 180, 45)
    if hough_lines is None or len(hough_lines) < 10:
        raise ValueError("Not enough edges detected. Make sure the whole pattern is visible, flat, and well-lit.")

    lines_vert, lines_hor = [], []
    for line in hough_lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = x0 + 1000 * (-b), y0 + 1000 * a
        x2, y2 = x0 - 1000 * (-b), y0 - 1000 * a
        l = np.cross([x1, y1, 1], [x2, y2, 1])
        l = l / l[2]
        (lines_vert if abs(a) > abs(b) else lines_hor).append(l)

    if not lines_vert or not lines_hor:
        raise ValueError("Couldn't find both horizontal and vertical pattern edges in this photo.")

    candidates = []
    for vl in lines_vert:
        for hl in lines_hor:
            c = np.cross(vl, hl)
            if abs(c[2]) < 1e-9:
                continue
            candidates.append(c[:2] / c[2])
    candidates = np.array(candidates, dtype=np.float32)
    if len(candidates) < N_CORNERS:
        raise ValueError("Not enough corner candidates found. Try a clearer, more head-on photo of the pattern.")

    kmeans = KMeans(n_clusters=N_CORNERS, n_init=10, random_state=0).fit(candidates)
    return kmeans.cluster_centers_


def sort_coords(corners: np.ndarray) -> np.ndarray:
    """Order 80 unsorted (x, y) corners into an 8-wide x 10-tall grid, row-major."""
    corners_h = np.hstack([corners, np.ones((len(corners), 1))])
    center = corners_h.mean(axis=0)
    A = B = C = D = None
    max_tl = max_tr = max_br = max_bl = -np.inf
    for point in corners_h:
        rel_x, rel_y, _ = point - center
        dist = float(np.linalg.norm(point - center))
        if rel_x <= 0 and rel_y <= 0 and dist > max_tl:
            max_tl, A = dist, point
        elif rel_x >= 0 and rel_y <= 0 and dist > max_tr:
            max_tr, B = dist, point
        elif rel_x >= 0 and rel_y >= 0 and dist > max_br:
            max_br, C = dist, point
        elif rel_x <= 0 and rel_y >= 0 and dist > max_bl:
            max_bl, D = dist, point
    if A is None or B is None or C is None or D is None:
        raise ValueError("Couldn't find the four extreme corners of the detected pattern.")

    dest = np.array([[0, 0], [RECT_W, 0], [RECT_W, RECT_H], [0, RECT_H]], dtype=np.float64)
    H = get_H_LLS(np.array([A[:2], B[:2], C[:2], D[:2]]), dest)
    transformed = H @ corners_h.T
    transformed = transformed / transformed[2]

    order_by_y = np.argsort(transformed[1])
    idx_by_row = order_by_y.reshape(GRID_ROWS, GRID_COLS)
    y_sorted = transformed[:, order_by_y]
    sorted_idx = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    for r in range(GRID_ROWS):
        start, end = r * GRID_COLS, (r + 1) * GRID_COLS
        sorted_idx[r] = idx_by_row[r, np.argsort(y_sorted[0, start:end])]

    return corners[sorted_idx.reshape(-1)]


_model_plane_coords: np.ndarray | None = None


def get_model_plane_coords() -> np.ndarray:
    global _model_plane_coords
    if _model_plane_coords is not None:
        return _model_plane_coords
    img = cv2.imread(str(PATTERN_PATH))
    corners = detect_corners(img)
    _model_plane_coords = sort_coords(corners)
    return _model_plane_coords


def _get_V_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j],
    ])


def get_img_conic(H_all: list[np.ndarray]) -> np.ndarray:
    V_all = np.zeros((len(H_all) * 2, 6))
    for i, H in enumerate(H_all):
        V_all[2 * i] = _get_V_ij(H, 0, 1)
        V_all[2 * i + 1] = _get_V_ij(H, 0, 0) - _get_V_ij(H, 1, 1)
    _, _, Vh = np.linalg.svd(V_all)
    b = Vh[-1]
    return np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])


def omega_to_K_inv(omega: np.ndarray) -> np.ndarray:
    denom = omega[0, 0] * omega[1, 1] - omega[0, 1] ** 2  # fixed: see module docstring
    y0 = (omega[0, 1] * omega[0, 2] - omega[0, 0] * omega[1, 2]) / denom
    lam = omega[2, 2] - (omega[0, 2] ** 2 + y0 * (omega[0, 1] * omega[0, 2] - omega[0, 0] * omega[1, 2])) / omega[0, 0]
    alpha_x = np.sqrt(lam / omega[0, 0])
    alpha_y = np.sqrt((lam * omega[0, 0]) / denom)
    s = -(omega[0, 1] * (alpha_x ** 2) * alpha_y) / lam
    x0 = s * y0 / alpha_y - (omega[0, 2] * alpha_x ** 2) / lam
    return np.array([[1 / alpha_x, 0, -x0 / alpha_x], [0, 1 / alpha_y, -y0 / alpha_y], [0, 0, 1]])


def get_projection_mat(H: np.ndarray, omega: np.ndarray):
    h1, h2, h3 = H[:, 0:1], H[:, 1:2], H[:, 2:3]
    K_inv = omega_to_K_inv(omega)
    scale = 1 / np.linalg.norm(K_inv @ h1)
    r1 = scale * K_inv @ h1
    r2 = scale * K_inv @ h2
    r3 = np.cross(r1.T, r2.T).reshape(3, 1)
    t = scale * K_inv @ h3
    R = np.hstack([r1, r2, r3])
    U, _, Vh = np.linalg.svd(R)
    R_cond = U @ Vh
    K = np.linalg.inv(K_inv)
    P = K @ np.hstack([R_cond, t])
    return P, K, R_cond, t


def projection_transform(P: np.ndarray, pts_src: np.ndarray) -> np.ndarray:
    pts_h = np.hstack([pts_src, np.zeros((len(pts_src), 1)), np.ones((len(pts_src), 1))])
    proj = P @ pts_h.T
    proj[:2] /= proj[2]
    return proj[:2].T


def lm_refine(H: np.ndarray, omega: np.ndarray, pts_src: np.ndarray, pts_target: np.ndarray):
    P0, K0, R0, t0 = get_projection_mat(H, omega)
    init = np.concatenate([K0.flatten(), R0.flatten(), t0.flatten()])

    def residual(params):
        K = params[0:9].reshape(3, 3)
        R = params[9:18].reshape(3, 3)
        t = params[18:21].reshape(3, 1)
        P = K @ np.hstack([R, t])
        return ((projection_transform(P, pts_src) - pts_target) ** 2).ravel()

    res = least_squares(residual, init, method="lm")
    K = res.x[0:9].reshape(3, 3)
    R = res.x[9:18].reshape(3, 3)
    t = res.x[18:21].reshape(3, 1)
    P = K @ np.hstack([R, t])
    return P, K, R, t


def draw_corners_overlay(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    out = img.copy()
    for x, y in corners:
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    return out


def draw_labeled_corners(img: np.ndarray, sorted_corners: np.ndarray) -> np.ndarray:
    out = img.copy()
    for i, (x, y) in enumerate(sorted_corners):
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
        cv2.putText(out, str(i + 1), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def draw_reprojection(img: np.ndarray, model_coords: np.ndarray, img_coords: np.ndarray, P: np.ndarray) -> np.ndarray:
    out = img.copy()
    pts_h = np.hstack([model_coords, np.zeros((len(model_coords), 1)), np.ones((len(model_coords), 1))])
    proj = P @ pts_h.T
    proj = proj[:2] / proj[2]
    for i in range(len(model_coords)):
        cv2.circle(out, (int(proj[0, i]), int(proj[1, i])), 4, (0, 0, 255), 1)
        cv2.circle(out, (int(img_coords[i][0]), int(img_coords[i][1])), 2, (0, 255, 0), -1)
    return out


def _draw_camera_frustum(ax, R: np.ndarray, C: np.ndarray, color, scale: float) -> None:
    """A small pyramid: apex at the camera center C, base a rectangle one
    `scale` unit along the camera's local +Z (its viewing direction, the
    convention `x_cam = R @ x_world + t` implies -- world points in front of
    the camera land at positive camera-frame Z). Reads as a recognizable
    "camera" glyph instead of an axis triad with no sense of which way it's
    pointing or how it relates to anything else in the scene.
    """
    hw, hh, dz = scale * 0.45, scale * 0.3, scale
    local = np.array([[-hw, -hh, dz], [hw, -hh, dz], [hw, hh, dz], [-hw, hh, dz]]).T
    world = ((R.T @ local) + C.reshape(3, 1)).T
    base = np.vstack([world, world[:1]])
    ax.plot(base[:, 0], base[:, 1], base[:, 2], color=color, linewidth=1.3)
    for corner in world:
        ax.plot([C[0], corner[0]], [C[1], corner[1]], [C[2], corner[2]], color=color, linewidth=1.0)
    ax.scatter(*C, color=color, s=20, depthshade=False)


def render_camera_poses(R_list: list[np.ndarray], t_list: list[np.ndarray]) -> np.ndarray:
    """Renders the calibration pattern as one real, checkered-textured plane
    at the world origin (Z=0 -- the plane every homography/extrinsic pair in
    `calibrate` is defined relative to), with each photo's recovered camera
    drawn as a small frustum pointed at it. The previous version drew a
    same-sized, randomly colored square floating at *each* camera's own
    position/orientation instead of the actual pattern, plus bare axis
    triads -- with no shared reference plane in the scene, there was no way
    to judge relative scale, distance, or which way a camera was actually
    pointed.
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    pattern_img = cv2.imread(str(PATTERN_PATH))
    pattern_rgb = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2RGB)
    tex_rows, tex_cols = 60, 46
    pattern_small = cv2.resize(pattern_rgb, (tex_cols, tex_rows), interpolation=cv2.INTER_AREA)
    facecolors = pattern_small.astype(np.float64) / 255.0
    X, Y = np.meshgrid(np.linspace(0, RECT_W, tex_cols + 1), np.linspace(0, RECT_H, tex_rows + 1))
    Z = np.zeros_like(X)
    ax.plot_surface(X, Y, Z, facecolors=facecolors, shade=False, rstride=1, cstride=1, antialiased=False, zorder=1)

    all_centers = np.array([(-R.T @ t).ravel() for R, t in zip(R_list, t_list)])
    plane_corners = np.array([[0, 0, 0], [RECT_W, 0, 0], [0, RECT_H, 0], [RECT_W, RECT_H, 0]])
    pts = np.vstack([all_centers, plane_corners])
    # Scale each camera frustum off the actual scene extent rather than a
    # fixed constant -- cameras a couple thousand units from the pattern (a
    # typical handheld-photo distance in these units) need a much bigger
    # glyph than 60 units to read as a camera at all, and a fixed size would
    # either vanish or dwarf the pattern depending on capture distance.
    scene_extent = float(np.max(np.ptp(pts, axis=0)))
    cam_scale = max(scene_extent * 0.09, 30.0)

    palette = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(R_list), 1)))
    for i, (R, t) in enumerate(zip(R_list, t_list)):
        C = all_centers[i]
        color = palette[i % len(palette)]
        _draw_camera_frustum(ax, R, C, color, cam_scale)
        ax.text(C[0], C[1], C[2], f"  {i + 1}", fontsize=8, color=color)

    margin = 80
    x_lo, x_hi = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    y_lo, y_hi = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    z_lo, z_hi = min(pts[:, 2].min() - margin, -20), max(pts[:, 2].max() + margin, 100)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_box_aspect((x_hi - x_lo, y_hi - y_lo, z_hi - z_lo))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Recovered camera poses relative to the calibration plane")
    ax.view_init(elev=28, azim=-60)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def calibrate(images: list[np.ndarray], progress=None) -> dict:
    if len(images) < 3:
        raise ValueError("Upload at least 3 photos of the calibration pattern from different angles.")

    images = [resize_for_demo(im) for im in images]
    model_coords = get_model_plane_coords()

    H_all, sorted_corners_all, corner_overlays, labeled_overlays = [], [], [], []
    for idx, img in enumerate(images):
        if progress:
            progress(f"Detecting corners in photo {idx + 1}/{len(images)}")
        raw_corners = detect_corners(img)
        corner_overlays.append(draw_corners_overlay(img, raw_corners))
        sorted_corners = sort_coords(raw_corners)
        sorted_corners_all.append(sorted_corners)
        labeled_overlays.append(draw_labeled_corners(img, sorted_corners))
        H_all.append(get_H_LLS(model_coords, sorted_corners))

    omega = get_img_conic(H_all)
    K_shared = np.linalg.inv(omega_to_K_inv(omega))

    reproj_before, reproj_after, R_list, t_list, K_list = [], [], [], [], []
    for idx, img in enumerate(images):
        if progress:
            progress(f"Refining pose {idx + 1}/{len(images)}")
        P0, K0, R0, t0 = get_projection_mat(H_all[idx], omega)
        P1, K1, R1, t1 = lm_refine(H_all[idx], omega, model_coords, sorted_corners_all[idx])
        reproj_before.append(draw_reprojection(img, model_coords, sorted_corners_all[idx], P0))
        reproj_after.append(draw_reprojection(img, model_coords, sorted_corners_all[idx], P1))
        R_list.append(R1)
        t_list.append(t1)
        K_list.append(K1)

    if progress:
        progress("Rendering camera poses")
    pose_plot = render_camera_poses(R_list, t_list)

    return {
        "corner_overlays": corner_overlays,
        "labeled_overlays": labeled_overlays,
        "reproj_before": reproj_before,
        "reproj_after": reproj_after,
        "K_shared": K_shared,
        "K_per_image": K_list,
        "pose_plot": pose_plot,
    }


def sample_image_paths(n: int = 8) -> list[str]:
    paths = sorted(SAMPLE_DIR.glob("Pic_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    step = max(1, len(paths) // n)
    return [str(p) for p in paths[::step][:n]]
