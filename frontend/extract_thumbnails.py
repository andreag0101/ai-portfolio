"""One-off script: extracts a representative image from each demo's bundled
precomputed result into public/thumbnails/<id>.jpg, for use as a project
card thumbnail on the site and in the GitHub profile README. Not part of
the app; safe to delete after running (or re-run if a precomputed sample
changes).
"""
import base64
import json
from pathlib import Path

from PIL import Image
import io

PUBLIC = Path(__file__).resolve().parent / "public"
OUT = PUBLIC / "thumbnails"
OUT.mkdir(exist_ok=True)

THUMB_SIZE = (480, 320)  # resize+crop to this, keeps cards visually consistent


def save_from_dataurl(data_url: str, out_name: str):
    b64 = data_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    # Crop to the target aspect ratio (center crop), then resize down.
    target_w, target_h = THUMB_SIZE
    target_ratio = target_w / target_h
    w, h = img.size
    ratio = w / h
    if ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize(THUMB_SIZE, Image.LANCZOS)
    img.save(OUT / out_name, quality=88)
    print(f"  wrote {out_name} ({img.size[0]}x{img.size[1]})")


SOURCES = [
    ("segmentation.json", "closingContour", "image-segmentation.jpg"),
    ("texture.json", "encodingVisual", "texture-classification.jpg"),
    ("calibration.json", "posePlot", "camera-calibration.jpg"),
    ("corners.json", "combined", "corner-matching.jpg"),
    ("disparity.json", "disparity", "stereo-disparity.jpg"),
    ("car-detection.json", "featureOverlay", "car-detection.jpg"),
]


def main():
    for filename, field, out_name in SOURCES:
        print(f"{filename} -> {field}")
        data = json.loads((PUBLIC / "precomputed" / filename).read_text())
        save_from_dataurl(data[field], out_name)

    print("face.json -> result.eigenfaces[0]")
    face = json.loads((PUBLIC / "precomputed" / "face.json").read_text())
    save_from_dataurl(face["result"]["eigenfaces"][0], "face-recognition.jpg")

    # Reuse existing curated images for demos that already have one.
    for src, out_name in [
        ("featured/rectify-after.jpg", "planar-rectification.jpg"),
        ("featured/panorama.jpg", "panorama.jpg"),
    ]:
        img = Image.open(PUBLIC / src).convert("RGB")
        w, h = img.size
        target_w, target_h = THUMB_SIZE
        target_ratio = target_w / target_h
        ratio = w / h
        if ratio > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            img = img.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            img = img.crop((0, top, w, top + new_h))
        img = img.resize(THUMB_SIZE, Image.LANCZOS)
        img.save(OUT / out_name, quality=88)
        print(f"  wrote {out_name} (from {src})")


if __name__ == "__main__":
    main()
