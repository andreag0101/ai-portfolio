from __future__ import annotations

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..cv_algorithms import texture as texture_algo
from ..utils.imaging import read_upload_as_bgr, to_data_url

router = APIRouter(prefix="/api/texture", tags=["texture"])


@router.get("/samples")
async def samples():
    images = [cv2.imread(str(p)) for p in texture_algo.SAMPLE_IMAGES]
    return {"samples": [to_data_url(im) for im in images if im is not None]}


@router.post("/classify")
async def classify(file: UploadFile = File(...)):
    img = await read_upload_as_bgr(file)
    try:
        result = texture_algo.classify(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    nearest = []
    for n in result["nearest"]:
        thumb = cv2.imread(n["path"])
        nearest.append({
            "label": n["label"],
            "distance": n["distance"],
            "thumbnail": to_data_url(cv2.resize(thumb, (96, 96))) if thumb is not None else None,
        })

    return {
        "predicted": result["predicted"],
        "confidences": result["confidences"],
        "nearest": nearest,
        "resizedInput": to_data_url(result["resized_input"]),
        "hueVisual": to_data_url(result["hue_visual"]),
        "encodingVisual": to_data_url(result["encoding_visual"]),
        "histogram": result["histogram"],
    }
