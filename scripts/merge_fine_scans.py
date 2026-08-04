"""Build merged RGB fine-scan TIFFs from Mn, Fe, and Cr fine-spot maps.

The coarse mosaic files are intentionally not read or modified here. Each
``fine_spot_###_<element>.tiff`` is a source plane for one
``fine_scan_###_merged_Mn_Fe_Cr.tiff`` RGB output. The merge operation is a
pixelwise maximum, which supports multiple source planes per region in a
future tiling workflow while leaving today's one-plane-per-region data
unchanged. Mn is the red channel, Fe green, and Cr blue.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile


DATA_DIR = Path(__file__).resolve().parents[1] / "fake_data"
ELEMENTS = ("Fe", "Mn", "Cr")
SOURCE_PATTERN = re.compile(r"fine_spot_(\d+)_(Fe|Mn|Cr)\.tiff$")
OLD_OUTPUT_PATTERN = re.compile(r"fine_scan_\d+_(Fe|Mn|Cr)\.tiff$")
MERGED_OUTPUT_PATTERN = re.compile(r"fine_scan_\d+_merged_(Fe_Cr_Mn|Mn_Fe_Cr)\.tiff$")


def merge_planes(paths: list[Path]) -> np.ndarray:
    """Merge same-sized source planes without changing their data type."""
    planes = [tifffile.imread(path) for path in paths]
    if not planes:
        raise ValueError("At least one source plane is required")
    if any(plane.shape != planes[0].shape for plane in planes[1:]):
        raise ValueError(f"Fine-scan source shapes do not match: {paths}")
    return np.maximum.reduce(planes).astype(planes[0].dtype, copy=False)


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Scale one element plane independently for an RGB display TIFF."""
    image = np.nan_to_num(image.astype(np.float32, copy=False))
    minimum, maximum = float(image.min()), float(image.max())
    if maximum == minimum:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image - minimum) * (255.0 / (maximum - minimum))).astype(np.uint8)


def main() -> None:
    # Delete only the obsolete, separate-element outputs made by an earlier
    # version of this script. Never touch source fine_spot maps or mosaics.
    for path in DATA_DIR.glob("fine_scan_*.tiff"):
        if OLD_OUTPUT_PATTERN.fullmatch(path.name) or MERGED_OUTPUT_PATTERN.fullmatch(path.name):
            path.unlink()

    sources: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in DATA_DIR.glob("fine_spot_*_*.tiff"):
        match = SOURCE_PATTERN.fullmatch(path.name)
        if match:
            scan_id, element = match.groups()
            sources[scan_id][element].append(path)

    if not sources:
        raise FileNotFoundError(f"No fine_spot TIFFs found in {DATA_DIR}")

    for scan_id, elements in sorted(sources.items()):
        missing = set(ELEMENTS) - set(elements)
        if missing:
            raise ValueError(f"Fine scan {scan_id} is missing: {', '.join(sorted(missing))}")
        element_planes = {element: merge_planes(elements[element]) for element in ELEMENTS}
        rgb = np.stack(
            [
                normalize_to_uint8(element_planes["Mn"]),
                normalize_to_uint8(element_planes["Fe"]),
                normalize_to_uint8(element_planes["Cr"]),
            ],
            axis=-1,
        )
        output = DATA_DIR / f"fine_scan_{scan_id}_merged_Mn_Fe_Cr.tiff"
        tifffile.imwrite(output, rgb, photometric="rgb")
        print(f"Wrote {output.name}")


if __name__ == "__main__":
    main()
