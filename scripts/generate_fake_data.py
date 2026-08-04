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
ELEMENTS = ("Fe", "Mn", "Cr")


def gaussian(x: np.ndarray, y: np.ndarray, x0: float, y0: float, sigma: float, amplitude: float) -> np.ndarray:
    """Return one circular XRF-like intensity feature."""
    return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def synthetic_element(x: np.ndarray, y: np.ndarray, element: str) -> np.ndarray:
    """Evaluate one continuous element map so coarse and fine scans align."""
    signal = np.full_like(x, 12.0, dtype=np.float64)
    signal += 1.8 * np.sin(x / 17.0) * np.cos(y / 23.0)

    # Each feature has a distinct response in Fe, Mn, and Cr. It lets the GUI
    # combine and select element maps while all channels share one coordinate frame.
    broad_features = (
        (78.0, 92.0, 3.0, {"Fe": 220.0, "Mn": 70.0, "Cr": 35.0}),
        (151.0, 61.0, 2.5, {"Fe": 95.0, "Mn": 185.0, "Cr": 55.0}),
        (207.0, 184.0, 3.8, {"Fe": 80.0, "Mn": 65.0, "Cr": 250.0}),
        (42.0, 205.0, 6.0, {"Fe": 80.0, "Mn": 20.0, "Cr": 40.0}),
        (224.0, 42.0, 5.0, {"Fe": 45.0, "Mn": 100.0, "Cr": 70.0}),
    )
    for x0, y0, sigma, amplitudes in broad_features:
        signal += gaussian(x, y, x0, y0, sigma, amplitudes[element])

    # Fine structure inside selected regions is suppressed by coarse sampling.
    fine_features = (
        (76.5, 91.2, 0.28, {"Fe": 170.0, "Mn": 35.0, "Cr": 20.0}),
        (79.3, 93.1, 0.36, {"Fe": 85.0, "Mn": 130.0, "Cr": 25.0}),
        (150.1, 60.0, 0.24, {"Fe": 25.0, "Mn": 150.0, "Cr": 45.0}),
        (152.0, 62.5, 0.32, {"Fe": 40.0, "Mn": 80.0, "Cr": 115.0}),
        (205.5, 183.2, 0.30, {"Fe": 65.0, "Mn": 40.0, "Cr": 185.0}),
        (208.4, 185.7, 0.42, {"Fe": 120.0, "Mn": 45.0, "Cr": 145.0}),
    )
    for x0, y0, sigma, amplitudes in fine_features:
        signal += gaussian(x, y, x0, y0, sigma, amplitudes[element])

    return signal.astype(np.float32)


def sample_image(
    element: str, size_px: int, pixel_size_um: float, center_um: tuple[float, float] | None = None
) -> np.ndarray:
    """Sample the scene at pixel centers, optionally around a fine-scan center."""
    if center_um is None:
        x0 = y0 = 0.0
    else:
        x0 = center_um[0] - FINE_FIELD_SIZE_UM / 2
        y0 = center_um[1] - FINE_FIELD_SIZE_UM / 2
    coordinates = (np.arange(size_px, dtype=np.float64) + 0.5) * pixel_size_um
    x, y = np.meshgrid(x0 + coordinates, y0 + coordinates)
    return synthetic_element(x, y, element)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    metadata: dict[str, object] = {
        "elements": list(ELEMENTS),
        "mosaic": {
            "files": [f"mosaic_{element}.tiff" for element in ELEMENTS],
            "shape_px": [MOSAIC_SIZE_PX, MOSAIC_SIZE_PX],
            "pixel_size_um": MOSAIC_PIXEL_SIZE_UM,
            "field_of_view_um": [MOSAIC_SIZE_PX * MOSAIC_PIXEL_SIZE_UM] * 2,
        },
        "fine_scans": [],
    }
    for element in ELEMENTS:
        tifffile.imwrite(
            OUTPUT_DIR / f"mosaic_{element}.tiff",
            sample_image(element, MOSAIC_SIZE_PX, MOSAIC_PIXEL_SIZE_UM),
        )
    for region in FINE_REGIONS:
        filenames = [f"fine_spot_{region['id']}_{element}.tiff" for element in ELEMENTS]
        for element, filename in zip(ELEMENTS, filenames, strict=True):
            tifffile.imwrite(
                OUTPUT_DIR / filename,
                sample_image(element, FINE_SIZE_PX, FINE_PIXEL_SIZE_UM, region["center_um"]),
            )
        metadata["fine_scans"].append(
            {
                "files": filenames,
                "shape_px": [FINE_SIZE_PX, FINE_SIZE_PX],
                "pixel_size_um": FINE_PIXEL_SIZE_UM,
                "field_of_view_um": [FINE_FIELD_SIZE_UM, FINE_FIELD_SIZE_UM],
                "center_in_mosaic_um": list(region["center_um"]),
            }
        )
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
