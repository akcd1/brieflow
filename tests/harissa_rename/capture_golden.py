"""Capture pre-rename watershed output as the golden reference.

Run this ONCE, against unrenamed code, before Task 4.
"""

from pathlib import Path

import numpy as np

from lib.shared.segment_watershed import segment_watershed

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def build_synthetic_image() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.integers(100, 200, size=(4, 256, 256), dtype=np.uint16)
    yy, xx = np.mgrid[0:256, 0:256]
    for cy, cx, radius in [(60, 60, 18), (60, 190, 15), (190, 60, 20), (190, 190, 16)]:
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2
        image[0][blob] = 3000
        image[1][blob] = 1500
    return image


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    labels = segment_watershed(
        data=build_synthetic_image(),
        nuclei_threshold=1000,
        nuclei_area_min=50,
        nuclei_area_max=5000,
        cell_threshold=500,
        cells=False,
    )
    np.save(GOLDEN_DIR / "watershed_labels.npy", labels)
    print(f"saved {GOLDEN_DIR / 'watershed_labels.npy'}, {len(np.unique(labels)) - 1} objects")


if __name__ == "__main__":
    main()
