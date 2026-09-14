# Computer Vision Portfolio

Interactive web demos of classical computer vision techniques, ported from
coursework (ECE 661, Purdue) into a from-scratch implementation of each
algorithm. Every demo can be stepped through stage by stage instead of only
showing a final result, every demo that takes an image (or pair/sequence of
images) has a one-click "use a sample" option, and every sample dataset the
demos need (calibration photos, a weather-photo training set, a face dataset,
car/non-car patches, and a curated sample input per demo) is bundled under
`backend/data/` so the whole thing runs standalone — no external files or
uploads required to try any of it.

## The 11 demos

- **Projective Geometry Playground** — point-line duality (cross products) as
  a live drag-the-triangle/drag-the-aim canvas.
- **Planar Rectification & Compositing** — automatic quadrilateral detection
  (Canny edges → contours → polygon approximation, filtered by
  rectangularity/angle sanity) + a 4-point DLT homography, to straighten a
  photographed plane or warp an image into a detected frame/screen.
- **Camera Calibration** — Zhang's method: a from-scratch Hough-line corner
  detector for a bespoke printable calibration pattern, per-photo
  homographies, the closed-form absolute-conic solution, and
  Levenberg-Marquardt refinement, stepped through corner detection → grid
  sorting → intrinsics → reprojection accuracy → 3D camera poses.
- **Corner Detection & Feature Matching** — a from-scratch Harris detector,
  SSD/NCC patch matching, plus an ORB+BFMatcher baseline for comparison.
- **Panorama Stitcher** — SIFT + a from-scratch RANSAC + Levenberg-Marquardt,
  stepped through interest points → inlier/outlier matching → the stitched
  result.
- **Interactive Image Segmentation** — iterative Otsu thresholding (RGB or
  texture) + morphological open/close + contours.
- **Texture-Based Weather Classification** — rotation-invariant circular LBP
  on the Hue channel, 1-nearest-neighbor against 922 bundled training photos.
- **Dense Stereo Depth Estimation** — census-transform window matching between
  a stereo pair, plus a step that unprojects the disparity map into an
  interactive, draggable 3D point cloud (three.js) colored from the source
  photo.
- **Face Recognition: Eigenfaces vs. Fisherfaces** — PCA and Fisher-LDA
  subspaces trained on 630 bundled photos of 30 people, 1-NN matching, a live
  held-out accuracy-vs-K chart, and the actual eigenface/fisherface basis
  images.
- **Car Detection: Haar Features + AdaBoost** — integral-image Haar-like
  features boosted with AdaBoost, Viola-Jones style, classifying cropped
  patches.

Several real bugs turned up in the original coursework code while porting it
(an axis mismatch in a `take_along_axis` call that silently corrupts AdaBoost
training, a sign/formula slip in the closed-form camera-intrinsics
extraction, a sort-direction bug that would greedily commit the *worst*
feature matches first) — each is documented with a before/after explanation
in the relevant module under `backend/app/cv_algorithms/`.

## Layout

- `backend/` — FastAPI app.
  - `app/cv_algorithms/` — the ported, vectorized algorithm code.
  - `app/routers/` — FastAPI endpoints exposing each algorithm.
  - `data/` — bundled sample datasets each demo needs to run (see below).
- `frontend/` — Vite + React + TypeScript + Tailwind UI.
  - `src/components/Stepper.tsx` — the shared step-through-the-pipeline UI
    every demo page uses.
  - `src/components/PointCloudViewer.tsx` — the three.js 3D point cloud viewer.

## Running locally

Two terminals — the backend and frontend run separately.

**Backend** (Python 3.8+; a conda/venv environment with the packages below):

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend**:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api/*` to the
backend on port 8000 (see `frontend/vite.config.ts`).

## Deployment

The backend does real per-request compute (SIFT, RANSAC, LM refinement,
OpenCV, scikit-learn), so it needs an actual running Python process, not
static/serverless hosting. Deployed as two separate services:

- **Backend** on [Render](https://render.com): a `render.yaml` blueprint at
  the repo root defines the web service (`rootDir: computer-vision/backend`,
  pinned to Python 3.11 via `.python-version`). Render → New → Blueprint →
  point it at this repo. The allowed CORS origin for the deployed frontend
  is set via the `ALLOWED_ORIGINS` env var (comma-separated), not hardcoded.
- **Frontend** on [Vercel](https://vercel.com): import the repo, set
  **Root Directory** to `computer-vision/frontend` (Vite preset
  auto-detected), and set the `VITE_API_BASE` env var to the deployed
  backend's URL plus `/api` (e.g. `https://<service>.onrender.com/api`) --
  in dev this defaults to `/api`, relying on the Vite proxy above, since
  frontend and backend aren't on the same origin in production.

After both are live, set `ALLOWED_ORIGINS` on the Render service to the
Vercel URL and redeploy. Render's free tier spins down after inactivity, so
the first request after idle time can take 30-60s to wake up.

## Data

`backend/data/` bundles what's needed for the demos that train, compare
against a reference set, or need a known-good sample input at runtime:

| Directory | Used by | Size |
|---|---|---|
| `data/texture/training/` | Texture classification (LBP training bank) | ~88 MB |
| `data/calibration/` | Camera calibration (pattern image + sample photo set) | ~1.5 MB |
| `data/face_recognition/{train,test}/` | Face recognition (PCA/LDA training + held-out accuracy) | ~46 MB |
| `data/car_detection/{train,test}/` | Car detection (AdaBoost training + held-out accuracy) | ~12 MB |
| `data/disparity/{left,right}.png` | Dense stereo (sample rectified pair) | ~1 MB |
| `data/homography/` | Planar rectification (a laptop-screen photo to straighten, plus a picture-frame + insert-image pair) | ~700 KB |
| `data/corners/{left,right}.jpg` | Corner matching (a two-viewpoint building photo pair, strong architectural corners) | ~300 KB |
| `data/panorama/{1..5}.jpg` | Panorama stitching (a 5-photo left-to-right sequence) | ~500 KB |
| `data/segmentation/` | Image segmentation (4 sample photos: dog, flower, tower, moon) | ~1 MB |
| `data/texture/samples/` | Texture classification (1 held-out test photo per weather class) | ~70 KB |

The first request to the texture, face, and car-detection endpoints trains or
builds a cache in memory (a few seconds to ~1 minute depending on the demo);
subsequent requests are fast.
