"""Generate deterministic, spatially matched synthetic XRF image data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "fake_data"
MOSAIC_SIZE_PX = 1040
MOSAIC_PIXEL_SIZE_UM = 0.25
FINE_SIZE_PX = 200
FINE_PIXEL_SIZE_UM = 0.05
FINE_FIELD_SIZE_UM = FINE_SIZE_PX * FINE_PIXEL_SIZE_UM

# Centers are in the mosaic's physical coordinate system (micrometres).
FINE_REGIONS = (
    {"id": "001", "center_um": (78.0, 92.0)},
    {"id": "002", "center_um": (151.0, 61.0)},
    {"id": "003", "center_um": (207.0, 184.0)},
)


def gaussian(x: np.ndarray, y: np.ndarray, x0: float, y0: float, sigma: float, amplitude: float) -> np.ndarray:
    """Return one circular XRF-like intensity feature."""
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def synthetic_fe(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a continuous sample so coarse and fine scans stay aligned."""
    signal = np.full_like(x, 12.0, dtype=np.float64)
    signal += 1.8 * np.sin(x / 17.0) * np.cos(y / 23.0)

    # Broad features are visible in both scan types.
    for x0, y0, sigma, amplitude in (
        (78.0, 92.0, 3.0, 220.0),
        (151.0, 61.0, 2.5, 185.0),
        (207.0, 184.0, 3.8, 250.0),
        (42.0, 205.0, 6.0, 80.0),
        (224.0, 42.0, 5.0, 100.0),
    ):
        signal += gaussian(x, y, x0, y0, sigma, amplitude)

    # Fine structure inside the selected regions is suppressed by coarse sampling.
    for x0, y0, sigma, amplitude in (
        (76.5, 91.2, 0.28, 170.0),
        (79.3, 93.1, 0.36, 130.0),
        (150.1, 60.0, 0.24, 150.0),
        (152.0, 62.5, 0.32, 115.0),
        (205.5, 183.2, 0.30, 185.0),
        (208.4, 185.7, 0.42, 145.0),
    ):
        signal += gaussian(x, y, x0, y0, sigma, amplitude)

    return signal.astype(np.float32)


def sample_image(size_px: int, pixel_size_um: float, center_um: tuple[float, float] | None = None) -> np.ndarray:
    """Sample the scene at pixel centers, optionally around a fine-scan center."""
    if center_um is None:
        x0 = y0 = 0.0
    else:
        x0 = center_um[0] - FINE_FIELD_SIZE_UM / 2
        y0 = center_um[1] - FINE_FIELD_SIZE_UM / 2
    coordinates = (np.arange(size_px, dtype=np.float64) + 0.5) * pixel_size_um
    x, y = np.meshgrid(x0 + coordinates, y0 + coordinates)
    return synthetic_fe(x, y)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    tifffile.imwrite(OUTPUT_DIR / "mosaic_Fe.tiff", sample_image(MOSAIC_SIZE_PX, MOSAIC_PIXEL_SIZE_UM))

    metadata: dict[str, object] = {
        "element": "Fe",
        "mosaic": {
            "file": "mosaic_Fe.tiff",
            "shape_px": [MOSAIC_SIZE_PX, MOSAIC_SIZE_PX],
            "pixel_size_um": MOSAIC_PIXEL_SIZE_UM,
            "field_of_view_um": [MOSAIC_SIZE_PX * MOSAIC_PIXEL_SIZE_UM] * 2,
        },
        "fine_scans": [],
    }
    for region in FINE_REGIONS:
        filename = f"fine_spot_{region['id']}_Fe.tiff"
        tifffile.imwrite(OUTPUT_DIR / filename, sample_image(FINE_SIZE_PX, FINE_PIXEL_SIZE_UM, region["center_um"]))
        metadata["fine_scans"].append(
            {
                "file": filename,
                "shape_px": [FINE_SIZE_PX, FINE_SIZE_PX],
                "pixel_size_um": FINE_PIXEL_SIZE_UM,
                "field_of_view_um": [FINE_FIELD_SIZE_UM, FINE_FIELD_SIZE_UM],
                "center_in_mosaic_um": list(region["center_um"]),
            }
        )
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
