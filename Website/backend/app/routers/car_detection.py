from __future__ import annotations

import cv2
from fastapi import APIRouter, UploadFile, File

from ..cv_algorithms import car_detection as cardet
from ..utils.imaging import read_upload_as_bgr, to_data_url

router = APIRouter(prefix="/api/car-detection", tags=["car-detection"])


@router.post("/classify")
async def classify(file: UploadFile = File(...)):
    img = await read_upload_as_bgr(file)
    result = cardet.run_classification(img)
    return {
        "isCar": result["is_car"],
        "score": result["score"],
        "threshold": result["threshold"],
        "numRounds": result["num_rounds"],
        "testAccuracy": result["test_accuracy"],
        "confusion": result["confusion"],
        "resizedPatch": to_data_url(result["resized_patch"]),
        "featureOverlay": to_data_url(result["feature_overlay"]),
    }


@router.get("/samples")
async def samples():
    paths = cardet.sample_paths(6)
    out = {}
    for key, plist in paths.items():
        out[key] = []
        for p in plist:
            img = cv2.imread(p)
            if img is not None:
                out[key].append(to_data_url(img))
    return out
