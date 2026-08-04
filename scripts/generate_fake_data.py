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
    {"id": "001", "center_um": (36.0, 42.0), "label": "isolated_Fe_particle"},
    {"id": "002", "center_um": (79.0, 80.0), "label": "partially_overlapping_Fe_Mn_pair"},
    {"id": "003", "center_um": (126.0, 53.0), "label": "three_element_overlap"},
    {"id": "004", "center_um": (178.0, 70.0), "label": "nearby_non_touching_Fe_Cr_pair"},
    {"id": "005", "center_um": (94.0, 177.0), "label": "large_Mn_Cr_particle"},
    {"id": "006", "center_um": (209.0, 207.0), "label": "large_isolated_Cr_particle"},
    {"id": "007", "center_um": (54.0, 132.0), "label": "isolated_Mn_particle"},
    {"id": "008", "center_um": (146.0, 145.0), "label": "partially_overlapping_Fe_Cr_pair"},
    {"id": "009", "center_um": (185.0, 121.0), "label": "nearby_non_touching_Mn_Cr_pair"},
    {"id": "010", "center_um": (229.0, 35.0), "label": "small_three_element_cluster"},
)
ELEMENTS = ("Fe", "Mn", "Cr")


def gaussian(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    angle_degrees: float,
    amplitude: float,
) -> np.ndarray:
    """Return one rotated, elliptical XRF-like particle."""
    angle = np.deg2rad(angle_degrees)
    dx, dy = x - x0, y - y0
    rotated_x = dx * np.cos(angle) + dy * np.sin(angle)
    rotated_y = -dx * np.sin(angle) + dy * np.cos(angle)
    return amplitude * np.exp(-(rotated_x**2 / (2 * sigma_x**2) + rotated_y**2 / (2 * sigma_y**2)))


def synthetic_element(x: np.ndarray, y: np.ndarray, element: str) -> np.ndarray:
    """Evaluate one continuous element map so coarse and fine scans align."""
    signal = np.full_like(x, 12.0, dtype=np.float64)
    signal += 1.8 * np.sin(x / 17.0) * np.cos(y / 23.0)

    # Sparse particles: isolated objects, partial overlaps, three-element
    # overlap, and a pair that is close but deliberately non-touching.
    particles = (
        # x, y, sigma_x, sigma_y, angle, Fe, Mn, Cr
        (36.0, 42.0, 1.4, 2.4, 15.0, 245.0, 0.0, 0.0),
        (76.8, 79.4, 2.5, 1.4, -20.0, 215.0, 0.0, 0.0),
        (80.7, 81.5, 2.0, 2.9, 35.0, 0.0, 210.0, 0.0),
        (124.5, 52.0, 2.6, 1.8, 10.0, 190.0, 0.0, 0.0),
        (127.0, 54.0, 2.0, 2.7, -35.0, 0.0, 185.0, 0.0),
        (126.0, 50.5, 1.6, 2.1, 45.0, 0.0, 0.0, 205.0),
        (174.0, 70.0, 1.5, 2.0, 0.0, 225.0, 0.0, 0.0),
        (182.0, 70.2, 1.6, 1.4, 20.0, 0.0, 0.0, 230.0),
        (94.0, 177.0, 4.6, 2.8, -10.0, 0.0, 210.0, 105.0),
        (98.0, 179.0, 2.3, 1.5, 40.0, 0.0, 75.0, 185.0),
        (209.0, 207.0, 5.5, 3.7, 25.0, 45.0, 0.0, 250.0),
        (54.0, 132.0, 2.4, 1.7, 55.0, 0.0, 235.0, 0.0),
        (144.5, 144.0, 3.1, 1.8, -15.0, 205.0, 0.0, 0.0),
        (147.8, 146.2, 2.0, 2.8, 35.0, 0.0, 0.0, 215.0),
        (181.0, 120.5, 1.9, 1.3, 10.0, 0.0, 205.0, 0.0),
        (189.0, 121.8, 1.5, 2.2, -30.0, 0.0, 0.0, 220.0),
        (228.0, 34.0, 1.8, 1.3, 5.0, 110.0, 45.0, 0.0),
        (230.2, 35.6, 1.4, 1.9, 40.0, 0.0, 100.0, 55.0),
        (228.8, 37.2, 1.2, 1.5, -25.0, 40.0, 0.0, 125.0),
        (48.0, 224.0, 2.0, 3.1, -40.0, 90.0, 145.0, 0.0),
    )
    element_index = {"Fe": 5, "Mn": 6, "Cr": 7}[element]
    for particle in particles:
        amplitude = particle[element_index]
        if amplitude:
            signal += gaussian(x, y, *particle[:5], amplitude)

    # Sub-micron details are visible in fine scans but mostly disappear in the
    # coarser mosaic. They make the selected scan genuinely more informative.
    fine_features = (
        (35.4, 41.1, 0.22, 0.35, 20.0, {"Fe": 130.0, "Mn": 0.0, "Cr": 0.0}),
        (77.5, 80.1, 0.30, 0.20, -25.0, {"Fe": 130.0, "Mn": 0.0, "Cr": 0.0}),
        (81.2, 82.0, 0.20, 0.36, 30.0, {"Fe": 0.0, "Mn": 140.0, "Cr": 0.0}),
        (125.2, 53.3, 0.25, 0.18, 0.0, {"Fe": 80.0, "Mn": 95.0, "Cr": 110.0}),
        (173.7, 69.2, 0.24, 0.32, 15.0, {"Fe": 145.0, "Mn": 0.0, "Cr": 0.0}),
        (182.5, 70.7, 0.28, 0.20, -10.0, {"Fe": 0.0, "Mn": 0.0, "Cr": 150.0}),
        (95.5, 176.0, 0.35, 0.22, 35.0, {"Fe": 0.0, "Mn": 115.0, "Cr": 80.0}),
        (207.5, 208.0, 0.32, 0.40, -30.0, {"Fe": 0.0, "Mn": 0.0, "Cr": 170.0}),
        (53.5, 132.3, 0.22, 0.34, 45.0, {"Fe": 0.0, "Mn": 155.0, "Cr": 0.0}),
        (145.0, 144.8, 0.36, 0.20, -15.0, {"Fe": 140.0, "Mn": 0.0, "Cr": 0.0}),
        (148.2, 146.6, 0.20, 0.32, 35.0, {"Fe": 0.0, "Mn": 0.0, "Cr": 160.0}),
        (180.7, 120.1, 0.25, 0.20, 10.0, {"Fe": 0.0, "Mn": 140.0, "Cr": 0.0}),
        (189.3, 122.4, 0.18, 0.30, -20.0, {"Fe": 0.0, "Mn": 0.0, "Cr": 165.0}),
        (229.0, 35.7, 0.22, 0.24, 0.0, {"Fe": 90.0, "Mn": 105.0, "Cr": 115.0}),
    )
    for x0, y0, sigma_x, sigma_y, angle, amplitudes in fine_features:
        if amplitudes[element]:
            signal += gaussian(x, y, x0, y0, sigma_x, sigma_y, angle, amplitudes[element])

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
    # Remove previous TIFFs owned by this generator before writing a fresh set.
    # Keep unrelated user files and downstream analysis outputs intact.
    for element in ELEMENTS:
        (OUTPUT_DIR / f"mosaic_{element}.tiff").unlink(missing_ok=True)
        for old_fine_scan in OUTPUT_DIR.glob(f"fine_spot_*_{element}.tiff"):
            old_fine_scan.unlink()

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
                "planned_region": region["label"],
            }
        )
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
