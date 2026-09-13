from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..cv_algorithms import corners as corner_algo
from ..utils.imaging import read_upload_as_bgr, to_data_url

router = APIRouter(prefix="/api/corners", tags=["corners"])


@router.post("/detect")
async def detect(file: UploadFile = File(...), sigma: float = Form(1.2)):
    sigma = max(0.5, min(sigma, 3.0))
    img = await read_upload_as_bgr(file)
    result = corner_algo.run_corner_detection(img, sigma)
    return {
        "input": to_data_url(result["input_resized"]),
        "overlay": to_data_url(result["overlay"]),
        "numCorners": result["num_corners"],
    }


@router.post("/match")
async def match(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    sigma: float = Form(1.2),
    distType: str = Form("SSD"),
):
    if distType not in ("SSD", "NCC"):
        raise HTTPException(status_code=400, detail="distType must be 'SSD' or 'NCC'")
    sigma = max(0.5, min(sigma, 3.0))
    img1 = await read_upload_as_bgr(file1)
    img2 = await read_upload_as_bgr(file2)
    result = corner_algo.run_corner_matching(img1, img2, sigma, distType)
    return {
        "cornersOverlay1": to_data_url(result["corners_overlay_1"]),
        "cornersOverlay2": to_data_url(result["corners_overlay_2"]),
        "combined": to_data_url(result["combined"]),
        "orbCombined": to_data_url(result["orb_combined"]),
        "numCorners1": result["num_corners_1"],
        "numCorners2": result["num_corners_2"],
        "numMatches": result["num_matches"],
    }
