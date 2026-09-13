from __future__ import annotations

from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..cv_algorithms import panorama as pano
from ..utils.imaging import read_upload_as_bgr, to_data_url

router = APIRouter(prefix="/api/panorama", tags=["panorama"])

MAX_IMAGES = 6


@router.post("")
async def stitch(files: List[UploadFile] = File(...), const: float = Form(10.0)):
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Upload at least two overlapping images.")
    if len(files) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_IMAGES} images.")

    images = [await read_upload_as_bgr(f) for f in files]

    try:
        result = pano.stitch_panorama(images, const=const)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "keypointOverlays": [to_data_url(im) for im in result["keypoint_overlays"]],
        "matchOverlays": [to_data_url(im) for im in result["match_overlays"]],
        "panorama": to_data_url(result["panorama"]),
    }
