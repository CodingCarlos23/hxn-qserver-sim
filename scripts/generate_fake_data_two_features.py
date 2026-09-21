"""Generate 5 synthetic two-feature XRF images for VLM bounding-box testing.

Each sample has exactly two distinct elliptical Gaussian blobs on a noisy
background, placed in well-separated zones so the VLM has an unambiguous task.
Ground-truth bounding boxes (fractional 0–1 coords, same schema as the Feature
Locator JSON) are saved alongside each TIFF so VLM output can be verified.

Image size: 512×512 px
  - Large enough that each blob covers a clearly visible region (~15–35% of
    the image width) without being the whole image.
  - No downsampling occurs when sent to the VLM (our 1024px cap is not hit).
  - Small enough to run quickly through the full GUI workflow.

Usage:
    pixi run python scripts/generate_fake_data_two_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "fake_data_two_features"
IMAGE_SIZE = 512
N_SAMPLES = 5
BOX_SIGMA_RADIUS = 3.0   # bounding box = center ± 3σ (covers ~99% of Gaussian mass)

SEEDS = [101, 202, 303, 404, 505]

# Four zone-pair layouts so the 5 samples have variety in blob placement.
# Each entry: (x_min_A, x_max_A, y_min_A, y_max_A, x_min_B, x_max_B, y_min_B, y_max_B)
# Zones are defined as fractions of image width/height.
ZONE_PAIRS = [
    (0.07, 0.43, 0.07, 0.43, 0.57, 0.93, 0.57, 0.93),  # top-left  + bottom-right
    (0.57, 0.93, 0.07, 0.43, 0.07, 0.43, 0.57, 0.93),  # top-right + bottom-left
    (0.07, 0.43, 0.07, 0.93, 0.57, 0.93, 0.07, 0.93),  # left      + right
    (0.07, 0.93, 0.07, 0.43, 0.07, 0.93, 0.57, 0.93),  # top       + bottom
]


def _gaussian(
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float, cy: float,
    sx: float, sy: float,
    angle_deg: float,
    amplitude: float,
) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    dx, dy = xx - cx, yy - cy
    rx = dx * np.cos(a) + dy * np.sin(a)
    ry = -dx * np.sin(a) + dy * np.cos(a)
    return amplitude * np.exp(-(rx ** 2 / (2 * sx ** 2) + ry ** 2 / (2 * sy ** 2)))


def _random_blob(rng: np.random.Generator, size: int,
                 xlo: float, xhi: float,
                 ylo: float, yhi: float) -> dict:
    margin = 60
    cx = float(rng.uniform(max(margin, xlo * size), min(size - margin, xhi * size)))
    cy = float(rng.uniform(max(margin, ylo * size), min(size - margin, yhi * size)))
    sx = float(rng.uniform(9, 16))
    sy = float(rng.uniform(9, 16))
    angle = float(rng.uniform(0, 180))
    amplitude = float(rng.uniform(130, 230))
    return dict(cx=cx, cy=cy, sx=sx, sy=sy, angle_deg=angle, amplitude=amplitude)


def _bbox(cx, cy, sx, sy, size, r=BOX_SIGMA_RADIUS):
    return (
        round(max(0.0, (cx - r * sx) / size), 4),
        round(max(0.0, (cy - r * sy) / size), 4),
        round(min(1.0, (cx + r * sx) / size), 4),
        round(min(1.0, (cy + r * sy) / size), 4),
    )


def generate_sample(seed: int, sample_idx: int) -> tuple[np.ndarray, dict]:  # returns uint16 TIFF array
    rng = np.random.default_rng(seed)
    size = IMAGE_SIZE
    xx, yy = np.meshgrid(np.arange(size, dtype=np.float32),
                         np.arange(size, dtype=np.float32))

    zones = ZONE_PAIRS[sample_idx % len(ZONE_PAIRS)]
    xloA, xhiA, yloA, yhiA, xloB, xhiB, yloB, yhiB = zones

    blobA = _random_blob(rng, size, xloA, xhiA, yloA, yhiA)
    blobB = _random_blob(rng, size, xloB, xhiB, yloB, yhiB)

    img = rng.normal(15.0, 3.5, (size, size)).astype(np.float32)
    img += _gaussian(xx, yy, **blobA).astype(np.float32)
    img += _gaussian(xx, yy, **blobB).astype(np.float32)
    img = np.clip(img, 0.0, None)
    mn, mx = img.min(), img.max()
    img = ((img - mn) / (mx - mn) * 65535).astype(np.uint16)

    x1A, y1A, x2A, y2A = _bbox(blobA["cx"], blobA["cy"], blobA["sx"], blobA["sy"], size)
    x1B, y1B, x2B, y2B = _bbox(blobB["cx"], blobB["cy"], blobB["sx"], blobB["sy"], size)

    ground_truth = {
        "sample": f"sample_{sample_idx:03d}",
        "seed": seed,
        "image_size_px": size,
        "features": [
            {
                "id": 1,
                "label": "feature_A",
                "x1": x1A, "y1": y1A, "x2": x2A, "y2": y2A,
                "center_px": [round(blobA["cx"], 1), round(blobA["cy"], 1)],
                "sigma_px": [round(blobA["sx"], 1), round(blobA["sy"], 1)],
                "amplitude": round(blobA["amplitude"], 1),
            },
            {
                "id": 2,
                "label": "feature_B",
                "x1": x1B, "y1": y1B, "x2": x2B, "y2": y2B,
                "center_px": [round(blobB["cx"], 1), round(blobB["cy"], 1)],
                "sigma_px": [round(blobB["sx"], 1), round(blobB["sy"], 1)],
                "amplitude": round(blobB["amplitude"], 1),
            },
        ],
        "total_count": 2,
    }
    return img, ground_truth


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for old in OUTPUT_DIR.glob("sample_*.tiff"):
        old.unlink()
    for old in OUTPUT_DIR.glob("sample_*_ground_truth.json"):
        old.unlink()

    print(f"Writing {N_SAMPLES} samples to {OUTPUT_DIR}/")
    for i, seed in enumerate(SEEDS, start=1):
        img, gt = generate_sample(seed, i)
        tiff_path = OUTPUT_DIR / f"sample_{i:03d}.tiff"
        gt_path = OUTPUT_DIR / f"sample_{i:03d}_ground_truth.json"
        tifffile.imwrite(tiff_path, img)
        gt_path.write_text(json.dumps(gt, indent=2) + "\n")
        fA, fB = gt["features"]
        print(
            f"  sample_{i:03d}.tiff  "
            f"A: center=({fA['center_px'][0]:.0f},{fA['center_px'][1]:.0f}) "
            f"box=[{fA['x1']},{fA['y1']} → {fA['x2']},{fA['y2']}]  "
            f"B: center=({fB['center_px'][0]:.0f},{fB['center_px'][1]:.0f}) "
            f"box=[{fB['x1']},{fB['y1']} → {fB['x2']},{fB['y2']}]"
        )
    print("Done.")


if __name__ == "__main__":
    main()
