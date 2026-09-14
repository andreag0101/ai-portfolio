import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import (
    segmentation,
    panorama,
    disparity,
    rectify,
    corners,
    texture,
    face,
    calibration,
    car_detection,
    rag,
)

# INFO is off by default in Python; without this, app.rag's per-request
# logging (see app/rag/logging_utils.py) silently never reaches stdout.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="ECE 661 Computer Vision Portfolio API")

# Local dev origins are always allowed; the deployed frontend's origin is
# added via the ALLOWED_ORIGINS env var (comma-separated) so it can be set
# per-environment without a code change/redeploy.
_default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_extra_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(segmentation.router)
app.include_router(panorama.router)
app.include_router(disparity.router)
app.include_router(rectify.router)
app.include_router(corners.router)
app.include_router(texture.router)
app.include_router(face.router)
app.include_router(calibration.router)
app.include_router(car_detection.router)
app.include_router(rag.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
