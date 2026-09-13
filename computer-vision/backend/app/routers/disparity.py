from __future__ import annotations

import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..cv_algorithms import disparity as disp
from ..utils.imaging import read_upload_as_bgr, to_data_url, array_to_base64

router = APIRouter(prefix="/api/disparity", tags=["disparity"])


@router.get("/sample")
async def sample():
    left = cv2.imread(str(disp.SAMPLE_LEFT))
    right = cv2.imread(str(disp.SAMPLE_RIGHT))
    return {"left": to_data_url(left), "right": to_data_url(right)}


def _point_cloud_json(positions, colors):
    return {
        "positions": array_to_base64(positions),
        "colors": array_to_base64(colors),
        "numPoints": int(positions.shape[0]),
    }


@router.post("")
async def compute(
    left: UploadFile = File(...),
    right: UploadFile = File(...),
    window: int = Form(9),
    d_max: int = Form(50),
):
    if window % 2 == 0 or not (3 <= window <= 19):
        raise HTTPException(status_code=400, detail="window must be an odd number between 3 and 19")
    d_max = max(5, min(d_max, 100))

    img_left = await read_upload_as_bgr(left)
    img_right = await read_upload_as_bgr(right)

    result = disp.run_disparity(img_left, img_right, window, d_max)

    return {
        "left": to_data_url(result["input_left_resized"]),
        "right": to_data_url(result["input_right_resized"]),
        "disparityGray": to_data_url(result["disparity_gray"]),
        "disparity": to_data_url(result["disparity_color"]),
        "filteredDisparity": to_data_url(result["filtered_disparity_visual"]),
        "filteredKeepFrac": result["filtered_keep_frac"],
        "sgbmDisparity": to_data_url(result["sgbm_disparity_visual"]),
        "sgbmKeepFrac": result["sgbm_keep_frac"],
        "pointCloud": _point_cloud_json(result["point_positions"], result["point_colors"]),
        "sgbmPointCloud": _point_cloud_json(result["sgbm_point_positions"], result["sgbm_point_colors"]),
    }
