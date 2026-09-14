from __future__ import annotations

import json
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from ..cv_algorithms import rectify as rect
from ..utils.imaging import read_upload_as_bgr, to_data_url

router = APIRouter(prefix="/api/rectify", tags=["rectify"])


@router.get("/sample")
async def sample():
    img = cv2.imread(str(rect.SAMPLE_STRAIGHTEN))
    return {"image": to_data_url(img)}


@router.get("/insert-sample")
async def insert_sample():
    dest = cv2.imread(str(rect.SAMPLE_FRAME))
    source = cv2.imread(str(rect.SAMPLE_INSERT_SOURCE))
    return {"dest": to_data_url(dest), "source": to_data_url(source)}


def _parse_quad(quad_json: Optional[str]) -> Optional[np.ndarray]:
    if not quad_json:
        return None
    try:
        pts = json.loads(quad_json)
        arr = np.array(pts, dtype=np.float32)
        assert arr.shape == (4, 2)
        return arr
    except Exception:
        raise HTTPException(status_code=400, detail="quad must be a JSON array of 4 [x, y] points")


@router.post("/candidates")
async def candidates(file: UploadFile = File(...)):
    img = await read_upload_as_bgr(file)
    resized, quads = rect.find_candidates(img, k=5)
    if not quads:
        raise HTTPException(status_code=422, detail=rect.NOT_FOUND_MSG)

    return {
        "image": to_data_url(resized),
        "candidates": [
            {
                "quad": q.tolist(),
                "overlay": to_data_url(rect.draw_quad_overlay(resized, q)),
            }
            for q in quads
        ],
    }


@router.post("/dewarp")
async def dewarp(file: UploadFile = File(...), quad: Optional[str] = Form(None)):
    img = await read_upload_as_bgr(file)
    chosen_quad = _parse_quad(quad)
    try:
        result = rect.run_dewarp(img, quad=chosen_quad)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "input": to_data_url(result["input_resized"]),
        "detectedQuad": to_data_url(result["detected_quad_overlay"]),
        "rectified": to_data_url(result["rectified"]),
    }


@router.post("/insert")
async def insert(dest: UploadFile = File(...), source: UploadFile = File(...), quad: Optional[str] = Form(None)):
    dest_img = await read_upload_as_bgr(dest)
    source_img = await read_upload_as_bgr(source)
    chosen_quad = _parse_quad(quad)
    try:
        result = rect.run_insert(dest_img, source_img, quad=chosen_quad)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "dest": to_data_url(result["dest_resized"]),
        "detectedQuad": to_data_url(result["detected_quad_overlay"]),
        "composited": to_data_url(result["composited"]),
    }
