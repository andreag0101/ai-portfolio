from __future__ import annotations

import cv2
from fastapi import APIRouter, UploadFile, File, HTTPException

from ..cv_algorithms import sparse_reconstruction as sr
from ..utils.imaging import read_upload_as_bgr, to_data_url, array_to_base64

router = APIRouter(prefix="/api/sparse-reconstruction", tags=["sparse-reconstruction"])


@router.get("/sample")
async def sample():
    left = cv2.imread(str(sr.SAMPLE_LEFT))
    right = cv2.imread(str(sr.SAMPLE_RIGHT))
    return {"left": to_data_url(left), "right": to_data_url(right)}


def _result_to_response(result):
    return {
        "left": to_data_url(result["left"]),
        "right": to_data_url(result["right"]),
        "numSiftMatches": result["num_sift_matches"],
        "numRansacInliers": result["num_ransac_inliers"],
        "keypointMatchOverlay": to_data_url(result["keypoint_match_overlay"]),
        "rectifiedLeft": to_data_url(result["rectified_left"]),
        "rectifiedRight": to_data_url(result["rectified_right"]),
        "edgesLeft": to_data_url(result["edges_left"]),
        "edgesRight": to_data_url(result["edges_right"]),
        "edgeMatchOverlay": to_data_url(result["edge_match_overlay"]),
        "numEdgeMatches": result["num_edge_matches"],
        "numReconstructed": result["num_reconstructed"],
        "medianParallaxDeg": result["median_parallax_deg"],
        "pointCloud": {
            "positions": array_to_base64(result["point_positions"]),
            "colors": array_to_base64(result["point_colors"]),
            "numPoints": int(result["point_positions"].shape[0]),
        },
    }


@router.post("")
async def compute(left: UploadFile = File(...), right: UploadFile = File(...)):
    img_left = await read_upload_as_bgr(left)
    img_right = await read_upload_as_bgr(right)
    try:
        result = sr.run_sparse_reconstruction(img_left, img_right)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return _result_to_response(result)
