from __future__ import annotations

import cv2
from fastapi import APIRouter, UploadFile, File, Form

from ..cv_algorithms import face as face_algo
from ..utils.imaging import read_upload_as_bgr, to_data_url

router = APIRouter(prefix="/api/face", tags=["face"])


@router.post("/classify")
async def classify(file: UploadFile = File(...), k: int = Form(10)):
    img = await read_upload_as_bgr(file)
    result = face_algo.classify(img, k)

    def with_thumb(match):
        thumb = cv2.imread(match["path"])
        return {
            "label": match["label"],
            "distance": match["distance"],
            "thumbnail": to_data_url(thumb) if thumb is not None else None,
        }

    return {
        "pca": with_thumb(result["pca"]),
        "lda": with_thumb(result["lda"]),
        "resizedInput": to_data_url(result["resized_input"]),
        "eigenfaces": [to_data_url(im) for im in result["eigenfaces"]],
        "fisherfaces": [to_data_url(im) for im in result["fisherfaces"]],
    }


@router.get("/accuracy-curve")
async def accuracy_curve():
    return face_algo.accuracy_curve()


@router.get("/samples")
async def samples():
    paths = face_algo.sample_test_paths(8)
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            out.append(to_data_url(img))
    return {"samples": out}
