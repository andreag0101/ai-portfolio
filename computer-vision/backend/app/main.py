from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import segmentation, panorama, disparity, rectify, corners, texture, face, calibration, car_detection

app = FastAPI(title="ECE 661 Computer Vision Portfolio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(segmentation.router)
app.include_router(panorama.router)
app.include_router(disparity.router)
app.include_router(rectify.router)
app.include_router(corners.router)
app.include_router(texture.router)
app.include_router(face.router)
app.include_router(calibration.router)
app.include_router(car_detection.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
