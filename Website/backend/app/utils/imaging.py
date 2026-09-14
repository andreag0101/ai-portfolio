from __future__ import annotations

import base64
import io

import numpy as np
import cv2
from PIL import Image, ImageOps
from fastapi import UploadFile, HTTPException


async def read_upload_as_bgr(file: UploadFile) -> np.ndarray:
    """Decode an upload to a BGR array, honoring EXIF orientation (phone photos
    commonly store pixel data in sensor orientation plus a rotation tag; ignoring
    it silently mangles any corner-order-sensitive algorithm, e.g. homography
    corner correspondences)."""
    data = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(data))
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb = np.array(pil_img.convert("RGB"))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {file.filename}")
    return img


def bgr_to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image")
    return buf.tobytes()


def gray_to_png_bytes(img_gray: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_gray)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image")
    return buf.tobytes()


def to_data_url(img: np.ndarray) -> str:
    """Encode a BGR or single-channel image as a PNG data: URL for direct <img> use."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image")
    encoded = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def array_to_base64(arr: np.ndarray) -> str:
    """Raw little-endian bytes of a numpy array, base64-encoded, for a JS
    typed array (Float32Array / Uint8Array) to read directly on the other end."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")
