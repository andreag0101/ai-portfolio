from __future__ import annotations

import cv2
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..cv_algorithms import segmentation as seg
from ..utils.imaging import read_upload_as_bgr, to_data_url

router = APIRouter(prefix="/api/segmentation", tags=["segmentation"])


@router.get("/samples")
async def samples():
    images = [cv2.imread(str(p)) for p in seg.SAMPLE_IMAGES]
    return {"samples": [to_data_url(im) for im in images if im is not None]}


@router.post("")
async def segment(
    file: UploadFile = File(...),
    mode: str = Form("rgb"),
    bins: int = Form(256),
    flip: bool = Form(False),
    iterations: int = Form(1),
):
    if mode not in ("rgb", "texture"):
        raise HTTPException(status_code=400, detail="mode must be 'rgb' or 'texture'")
    bins = max(8, min(bins, 256))
    iterations = max(0, min(iterations, 5))

    img = await read_upload_as_bgr(file)
    result = seg.run_segmentation(img, mode, bins, flip, iterations)

    return {
        "input": to_data_url(result["input_resized"]),
        "mask": to_data_url(result["mask"]),
        "maskBeforeIteration": to_data_url(result["mask_before_iteration"]),
        "contour": to_data_url(result["contour"]),
        "openingMask": to_data_url(result["opening_mask"]),
        "openingContour": to_data_url(result["opening_contour"]),
        "closingMask": to_data_url(result["closing_mask"]),
        "closingContour": to_data_url(result["closing_contour"]),
    }
