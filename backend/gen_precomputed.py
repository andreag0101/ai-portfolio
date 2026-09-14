"""Regenerates frontend/public/precomputed/*.json -- the static, pre-computed
sample results each demo page shows by default (so the site works even if
the separately-hosted backend is asleep/unreachable; see getPrecomputed in
frontend/src/lib/api.ts). Drives a running local dev backend through the
exact same request flow each page's "Use sample" button uses, so the output
is byte-identical to what that button would produce.

Run whenever a bundled sample dataset or an algorithm changes:

    conda activate ece661   # or your venv
    uvicorn app.main:app --reload --port 8000  # in another terminal
    python gen_precomputed.py                  # regenerates all of them
    python gen_precomputed.py calibration       # or just one, by name

Not imported by the app; safe to leave in the repo.
"""
import base64
import json
import sys
from pathlib import Path

import requests

BASE = "http://localhost:8000/api"
OUT_DIR = Path(__file__).resolve().parents[1] / "frontend" / "public" / "precomputed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dataurl_to_bytes(data_url: str) -> bytes:
    b64 = data_url.split(",", 1)[1]
    return base64.b64decode(b64)


def save(name: str, payload: dict):
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload))
    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)")


def gen_segmentation():
    print("segmentation...")
    samples = requests.get(f"{BASE}/segmentation/samples").json()["samples"]
    files = {"file": ("sample.jpg", dataurl_to_bytes(samples[0]), "image/jpeg")}
    data = {"mode": "rgb", "bins": "256", "flip": "False", "iterations": "1"}
    r = requests.post(f"{BASE}/segmentation", files=files, data=data)
    r.raise_for_status()
    save("segmentation", r.json())


def gen_panorama():
    print("panorama...")
    images = requests.get(f"{BASE}/panorama/sample").json()["images"]
    files = [("files", (f"{i}.jpg", dataurl_to_bytes(im), "image/jpeg")) for i, im in enumerate(images)]
    data = {"const": "5"}
    r = requests.post(f"{BASE}/panorama", files=files, data=data)
    r.raise_for_status()
    save("panorama", r.json())


def gen_disparity():
    print("disparity...")
    sample = requests.get(f"{BASE}/disparity/sample").json()
    files = {
        "left": ("left.png", dataurl_to_bytes(sample["left"]), "image/png"),
        "right": ("right.png", dataurl_to_bytes(sample["right"]), "image/png"),
    }
    data = {"window": "9", "d_max": "50"}
    r = requests.post(f"{BASE}/disparity", files=files, data=data)
    r.raise_for_status()
    save("disparity", r.json())


def gen_homography_straighten():
    print("homography-straighten...")
    sample = requests.get(f"{BASE}/rectify/sample").json()
    img_bytes = dataurl_to_bytes(sample["image"])
    cand = requests.post(f"{BASE}/rectify/candidates", files={"file": ("sample.jpg", img_bytes, "image/jpeg")})
    cand.raise_for_status()
    quad = cand.json()["candidates"][0]["quad"]
    r = requests.post(
        f"{BASE}/rectify/dewarp",
        files={"file": ("sample.jpg", img_bytes, "image/jpeg")},
        data={"quad": json.dumps(quad)},
    )
    r.raise_for_status()
    save("homography-straighten", r.json())


def gen_homography_insert():
    print("homography-insert...")
    sample = requests.get(f"{BASE}/rectify/insert-sample").json()
    dest_bytes = dataurl_to_bytes(sample["dest"])
    source_bytes = dataurl_to_bytes(sample["source"])
    cand = requests.post(f"{BASE}/rectify/candidates", files={"file": ("dest.jpg", dest_bytes, "image/jpeg")})
    cand.raise_for_status()
    quad = cand.json()["candidates"][0]["quad"]
    r = requests.post(
        f"{BASE}/rectify/insert",
        files={
            "dest": ("dest.jpg", dest_bytes, "image/jpeg"),
            "source": ("source.jpg", source_bytes, "image/jpeg"),
        },
        data={"quad": json.dumps(quad)},
    )
    r.raise_for_status()
    save("homography-insert", r.json())


def gen_corners():
    print("corners...")
    sample = requests.get(f"{BASE}/corners/sample").json()
    files = {
        "file1": ("1.jpg", dataurl_to_bytes(sample["file1"]), "image/jpeg"),
        "file2": ("2.jpg", dataurl_to_bytes(sample["file2"]), "image/jpeg"),
    }
    data = {"sigma": "1.2", "distType": "SSD"}
    r = requests.post(f"{BASE}/corners/match", files=files, data=data)
    r.raise_for_status()
    save("corners", r.json())


def gen_texture():
    print("texture...")
    samples = requests.get(f"{BASE}/texture/samples").json()["samples"]
    files = {"file": ("sample.jpg", dataurl_to_bytes(samples[0]), "image/jpeg")}
    r = requests.post(f"{BASE}/texture/classify", files=files)
    r.raise_for_status()
    save("texture", r.json())


def gen_face():
    print("face...")
    samples = requests.get(f"{BASE}/face/samples").json()["samples"]
    files = {"file": ("sample.jpg", dataurl_to_bytes(samples[0]), "image/jpeg")}
    data = {"k": "10"}
    r = requests.post(f"{BASE}/face/classify", files=files, data=data)
    r.raise_for_status()
    result = r.json()
    curve = requests.get(f"{BASE}/face/accuracy-curve")
    curve.raise_for_status()
    save("face", {"result": result, "accuracyCurve": curve.json()})


def gen_car_detection():
    print("car-detection...")
    samples = requests.get(f"{BASE}/car-detection/samples").json()
    files = {"file": ("sample.jpg", dataurl_to_bytes(samples["positive"][0]), "image/jpeg")}
    r = requests.post(f"{BASE}/car-detection/classify", files=files)
    r.raise_for_status()
    save("car-detection", r.json())


def gen_calibration():
    print("calibration (this one is slow)...")
    r = requests.post(f"{BASE}/calibration/run-sample")
    r.raise_for_status()
    save("calibration", r.json())


if __name__ == "__main__":
    which = sys.argv[1:] or None
    generators = {
        "segmentation": gen_segmentation,
        "panorama": gen_panorama,
        "disparity": gen_disparity,
        "homography-straighten": gen_homography_straighten,
        "homography-insert": gen_homography_insert,
        "corners": gen_corners,
        "texture": gen_texture,
        "face": gen_face,
        "car-detection": gen_car_detection,
        "calibration": gen_calibration,
    }
    for name, fn in generators.items():
        if which and name not in which:
            continue
        fn()
    print("done")
