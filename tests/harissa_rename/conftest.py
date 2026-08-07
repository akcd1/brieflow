"""Fixtures for the harissa vacuole-rename characterization tests."""

from pathlib import Path

import numpy as np
import pytest

WORKFLOW_ROOT = Path(__file__).resolve().parents[2] / "workflow"
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


@pytest.fixture(scope="session")
def synthetic_image() -> np.ndarray:
    """A deterministic 4-channel image with blob-like objects.

    Channel 0 stands in for the object-marking channel, channel 1 for a
    cytoplasmic stain. Values are uint16 to match real acquisition data.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(100, 200, size=(4, 256, 256), dtype=np.uint16)

    yy, xx = np.mgrid[0:256, 0:256]
    for cy, cx, radius in [(60, 60, 18), (60, 190, 15), (190, 60, 20), (190, 190, 16)]:
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2
        image[0][blob] = 3000
        image[1][blob] = 1500

    return image


@pytest.fixture(scope="session")
def source_text():
    """Read a workflow source file as text, for string-preservation checks."""

    def _read(relative_path: str) -> str:
        return (WORKFLOW_ROOT / relative_path).read_text()

    return _read
