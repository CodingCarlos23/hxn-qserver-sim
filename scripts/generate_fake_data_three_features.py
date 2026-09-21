"""Generate 5 synthetic three-feature XRF images for VLM bounding-box testing.

Each sample has exactly three distinct elliptical Gaussian blobs on a noisy
background, placed in well-separated zones so the VLM has an unambiguous task.
Ground-truth bounding boxes (fractional 0–1 coords, same schema as the Feature
Locator JSON) are saved alongside each TIFF so VLM output can be verified.

Image size: 512×512 px
  - Large enough that each blob covers a clearly visible region without
    dominating the whole image.
  - No downsampling occurs when sent to the VLM (our 1024px cap is not hit).
  - Small enough to run quickly through the full GUI workflow.

Usage:
    pixi run python scripts/generate_fake_data_three_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "fake_data_three_features"
IMAGE_SIZE = 512
N_SAMPLES = 5
BOX_SIGMA_RADIUS = 3.0   # bounding box = center ± 3σ (covers ~99% of Gaussian mass)

SEEDS = [101, 202, 303, 404, 505]

FEATURE_LABELS = ["feature_A", "feature_B", "feature_C"]

# Four zone-set layouts, each with 3 zones as (xlo, xhi, ylo, yhi) fractions.
ZONE_SETS = [
    # Layout 0: top-left + top-right + bottom-centre
    [(0.05, 0.43, 0.05, 0.45), (0.57, 0.95, 0.05, 0.45), (0.20, 0.80, 0.55, 0.95)],
    # Layout 1: top-centre + bottom-left + bottom-right
    [(0.20, 0.80, 0.05, 0.43), (0.05, 0.43, 0.57, 0.95), (0.57, 0.95, 0.57, 0.95)],
    # Layout 2: left + centre + right (horizontal strip)
    [(0.05, 0.38, 0.05, 0.95), (0.31, 0.69, 0.05, 0.95), (0.62, 0.95, 0.05, 0.95)],
    # Layout 3: top-left + bottom-right + centre-offset
    [(0.05, 0.40, 0.05, 0.43), (0.60, 0.95, 0.57, 0.95), (0.25, 0.75, 0.20, 0.80)],
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


def generate_sample(seed: int, sample_idx: int) -> tuple[np.ndarray, dict]:  # returns uint16 array
    rng = np.random.default_rng(seed)
    size = IMAGE_SIZE
    xx, yy = np.meshgrid(np.arange(size, dtype=np.float32),
                         np.arange(size, dtype=np.float32))

    zones = ZONE_SETS[sample_idx % len(ZONE_SETS)]
    blobs = [_random_blob(rng, size, *z) for z in zones]

    img = rng.normal(15.0, 3.5, (size, size)).astype(np.float32)
    for blob in blobs:
        img += _gaussian(xx, yy, **blob).astype(np.float32)
    img = np.clip(img, 0.0, None)
    mn, mx = img.min(), img.max()
    img = ((img - mn) / (mx - mn) * 65535).astype(np.uint16)

    features = []
    for i, (blob, label) in enumerate(zip(blobs, FEATURE_LABELS), start=1):
        x1, y1, x2, y2 = _bbox(blob["cx"], blob["cy"], blob["sx"], blob["sy"], size)
        features.append({
            "id": i,
            "label": label,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "center_px": [round(blob["cx"], 1), round(blob["cy"], 1)],
            "sigma_px": [round(blob["sx"], 1), round(blob["sy"], 1)],
            "amplitude": round(blob["amplitude"], 1),
        })

    ground_truth = {
        "sample": f"sample_{sample_idx:03d}",
        "seed": seed,
        "image_size_px": size,
        "features": features,
        "total_count": 3,
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
        centers = "  ".join(
            f"{f['label'][-1]}:({f['center_px'][0]:.0f},{f['center_px'][1]:.0f})"
            for f in gt["features"]
        )
        print(f"  sample_{i:03d}.tiff  {centers}")
    print("Done.")


if __name__ == "__main__":
    main()
