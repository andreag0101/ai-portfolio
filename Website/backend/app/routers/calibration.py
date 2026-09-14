from __future__ import annotations

from typing import List

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..cv_algorithms import calibration as calib
from ..utils.imaging import read_upload_as_bgr, to_data_url

router = APIRouter(prefix="/api/calibration", tags=["calibration"])

MAX_IMAGES = 12


def _k_to_dict(K):
    return {
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "skew": float(K[0, 1]),
    }


@router.get("/pattern")
async def pattern():
    img = cv2.imread(str(calib.PATTERN_PATH))
    return {"pattern": to_data_url(img)}


@router.get("/samples")
async def samples():
    paths = calib.sample_image_paths(8)
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            out.append(to_data_url(cv2.resize(img, (160, 120))))
    return {"samples": out}


def _result_to_response(result):
    return {
        "cornerOverlays": [to_data_url(im) for im in result["corner_overlays"]],
        "labeledOverlays": [to_data_url(im) for im in result["labeled_overlays"]],
        "reprojBefore": [to_data_url(im) for im in result["reproj_before"]],
        "reprojAfter": [to_data_url(im) for im in result["reproj_after"]],
        "intrinsics": _k_to_dict(result["K_shared"]),
        "posePlot": to_data_url(result["pose_plot"]),
    }


@router.post("")
async def run_calibration(files: List[UploadFile] = File(...)):
    if len(files) < 3:
        raise HTTPException(status_code=400, detail="Upload at least 3 photos of the calibration pattern.")
    if len(files) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_IMAGES} images.")

    images = [await read_upload_as_bgr(f) for f in files]
    try:
        result = calib.calibrate(images)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return _result_to_response(result)


@router.post("/run-sample")
async def run_sample_calibration():
    paths = calib.sample_image_paths(8)
    images = [cv2.imread(p) for p in paths]
    images = [im for im in images if im is not None]
    try:
        result = calib.calibrate(images)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return _result_to_response(result)
