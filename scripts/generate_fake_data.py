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
ELEMENTS = ("Fe", "Mn", "Cr")
COMBINATIONS = (
    ("Fe",),
    ("Mn",),
    ("Cr",),
    ("Fe", "Mn"),
    ("Fe", "Cr"),
    ("Mn", "Cr"),
    ("Fe", "Mn", "Cr"),
)
LAYOUT_SEED = 20260804


def scattered_centers(count: int) -> tuple[tuple[float, float], ...]:
    """Create a repeatable, non-grid layout with generous space between AOIs."""
    rng = np.random.default_rng(LAYOUT_SEED)
    centers: list[tuple[float, float]] = []
    minimum_spacing_um = 28.0
    while len(centers) < count:
        candidate = tuple(float(value) for value in rng.uniform(15.0, 245.0, size=2))
        if all(np.hypot(candidate[0] - x, candidate[1] - y) >= minimum_spacing_um for x, y in centers):
            centers.append(candidate)
    return tuple(centers)


REGION_COMBINATIONS = COMBINATIONS * 2 + (("Fe",), ("Mn",), ("Cr",), ("Fe", "Mn"), ("Fe", "Cr"), ("Mn", "Cr"))
REGION_CENTERS = scattered_centers(len(REGION_COMBINATIONS))
FINE_REGIONS = tuple(
    {
        "id": f"{index:03}",
        "center_um": center,
        "elements": elements,
        "label": "_".join(elements) + ("_particle" if len(elements) == 1 else "_cluster"),
    }
    for index, (center, elements) in enumerate(zip(REGION_CENTERS, REGION_COMBINATIONS, strict=True), start=1)
)


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

    # Each selected region receives one or more element-specific particles.
    # Pairs alternate between partial overlap and a close non-touching layout.
    offsets_by_count = {
        1: ((0.0, 0.0),),
        2: ((-1.2, -0.6), (1.2, 0.6)),
        3: ((-1.3, -0.8), (1.3, -0.5), (0.0, 1.3)),
    }
    for region_index, region in enumerate(FINE_REGIONS, start=1):
        components = offsets_by_count[len(region["elements"])]
        if len(components) == 2 and region_index % 2 == 0:
            components = ((-2.3, 0.0), (2.3, 0.0))
        center_x, center_y = region["center_um"]
        for element_index, (particle_element, (offset_x, offset_y)) in enumerate(zip(region["elements"], components, strict=True)):
            if particle_element != element:
                continue
            sigma_x = 1.2 + 0.35 * ((region_index + element_index) % 5)
            sigma_y = 1.3 + 0.30 * ((2 * region_index + element_index) % 5)
            angle = float((region_index * 29 + element_index * 47) % 180)
            amplitude = 180.0 + 15.0 * ((region_index + 2 * element_index) % 5)
            particle_x, particle_y = center_x + offset_x, center_y + offset_y
            signal += gaussian(x, y, particle_x, particle_y, sigma_x, sigma_y, angle, amplitude)
            signal += gaussian(x, y, particle_x + 0.35, particle_y - 0.25, 0.22, 0.30, angle + 20.0, 95.0)

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
                "elements": list(region["elements"]),
            }
        )
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
