"""Unit tests for the axis-specific edge-offset cell filter in the aggregate module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make the workflow's `lib` package importable without installing the package
WORKFLOW_LIB_DIR = Path(__file__).resolve().parents[1] / "workflow"
sys.path.insert(0, str(WORKFLOW_LIB_DIR))

from lib.aggregate.filter import edge_offset_filter


def _data():
    """tile_size=100. cell_bounds_* = (min_row, min_col, max_row, max_col).

    A: top edge at pixel 0, offset 0           -> drop (d_top=0 not > 0)
    B: d_top=3 <= |offset_y|=5                  -> drop
    C: centered, clears all edges              -> keep
    D: d_left=2 <= |offset_x|=3                 -> drop
    E: d_top=2 > |offset_y|=1 (axis clears) but |offset_x|=10 large on the
       far col edges (d_left=40, d_right=50 both clear) -> keep.
       This is the discriminating case: a blanket max(|ox|,|oy|)=10 margin
       would drop E, but the axis-specific rule keeps it.
    F: NaN offset                              -> drop (cannot verify)
    """
    meta = pd.DataFrame(
        [
            dict(
                cell_bounds_0=0,
                cell_bounds_1=50,
                cell_bounds_2=10,
                cell_bounds_3=60,
                offset_y=0,
                offset_x=0,
                tag="A",
            ),
            dict(
                cell_bounds_0=3,
                cell_bounds_1=50,
                cell_bounds_2=13,
                cell_bounds_3=60,
                offset_y=5,
                offset_x=2,
                tag="B",
            ),
            dict(
                cell_bounds_0=40,
                cell_bounds_1=40,
                cell_bounds_2=60,
                cell_bounds_3=60,
                offset_y=5,
                offset_x=5,
                tag="C",
            ),
            dict(
                cell_bounds_0=50,
                cell_bounds_1=2,
                cell_bounds_2=60,
                cell_bounds_3=12,
                offset_y=1,
                offset_x=3,
                tag="D",
            ),
            dict(
                cell_bounds_0=2,
                cell_bounds_1=40,
                cell_bounds_2=12,
                cell_bounds_3=50,
                offset_y=1,
                offset_x=10,
                tag="E",
            ),
            dict(
                cell_bounds_0=40,
                cell_bounds_1=40,
                cell_bounds_2=60,
                cell_bounds_3=60,
                offset_y=np.nan,
                offset_x=np.nan,
                tag="F",
            ),
        ]
    )
    feats = pd.DataFrame({"fval": [10, 20, 30, 40, 50, 60]})
    return meta, feats


def test_keeps_only_cells_clearing_their_axis_offset():
    meta, feats = _data()
    m, f = edge_offset_filter(meta, feats, tile_size=100)
    assert m["tag"].tolist() == ["C", "E"]
    # features stay row-aligned to the kept metadata
    assert f["fval"].tolist() == [30, 50]
    assert len(m) == len(f) == 2


def test_nan_offset_is_dropped():
    meta, feats = _data()
    m, _ = edge_offset_filter(meta, feats, tile_size=100)
    assert "F" not in m["tag"].tolist()


def test_signed_offset_uses_absolute_value():
    # A large negative offset must drop the cell just like a large positive one.
    meta = pd.DataFrame(
        [
            dict(
                cell_bounds_0=5,
                cell_bounds_1=50,
                cell_bounds_2=15,
                cell_bounds_3=60,
                offset_y=-10,
                offset_x=0,
                tag="neg",
            ),
        ]
    )
    feats = pd.DataFrame({"fval": [1]})
    m, f = edge_offset_filter(meta, feats, tile_size=100)
    assert len(m) == 0  # d_top=5 <= |−10| -> dropped
    assert len(f) == 0


def test_alignment_robust_to_duplicate_index():
    meta, feats = _data()
    meta.index = [0] * len(meta)
    feats.index = [0] * len(feats)
    m, f = edge_offset_filter(meta, feats, tile_size=100)
    assert m["tag"].tolist() == ["C", "E"]
    assert f["fval"].tolist() == [30, 50]


def test_missing_required_column_raises():
    meta, feats = _data()
    with pytest.raises(ValueError):
        edge_offset_filter(meta.drop(columns=["offset_x"]), feats, tile_size=100)
