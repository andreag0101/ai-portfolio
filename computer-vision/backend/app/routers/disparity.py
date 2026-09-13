from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..cv_algorithms import disparity as disp
from ..utils.imaging import read_upload_as_bgr, to_data_url, array_to_base64

router = APIRouter(prefix="/api/disparity", tags=["disparity"])


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
        "pointCloud": {
            "positions": array_to_base64(result["point_positions"]),
            "colors": array_to_base64(result["point_colors"]),
            "numPoints": int(result["point_positions"].shape[0]),
        },
    }
