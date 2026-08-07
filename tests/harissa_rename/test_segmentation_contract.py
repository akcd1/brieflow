"""Proves the vacuole rename does not change segmentation behavior.

Written against UNRENAMED code, then re-run after each rename step. The
watershed test is the behavioral anchor: it runs real segmentation on CPU
in about a second, with no model download.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).resolve().parent / "golden" / "watershed_labels.npy"

# Markers that identify a line as Cellpose nomenclature rather than our own.
CELLPOSE_CONTEXT = ("model_type", "size_model_path", "cellpose_model", "CELLPOSE_4X")

SEGMENTATION_LIBS = [
    "lib/shared/segment_cellpose.py",
    "lib/shared/segment_stardist.py",
    "lib/shared/segment_watershed.py",
]


@pytest.mark.parametrize("relative_path", SEGMENTATION_LIBS)
def test_remaining_nuclei_is_only_cellpose_nomenclature(relative_path, source_text):
    """The invariant: any bare 'nuclei' left must be a Cellpose model name.

    This replaces a line-number exception list. It stays true as the code
    evolves, so a future edit cannot quietly reintroduce the ambiguity
    between Cellpose's vocabulary and our object name.
    """
    offenders = []
    for lineno, line in enumerate(source_text(relative_path).splitlines(), start=1):
        if "nuclei" not in line.lower():
            continue
        if any(marker in line for marker in CELLPOSE_CONTEXT):
            continue
        offenders.append(f"{relative_path}:{lineno}: {line.strip()}")

    assert not offenders, "non-Cellpose 'nuclei' survived the rename:\n" + "\n".join(offenders)


def test_cellpose_model_names_intact(source_text):
    """Cellpose's own model strings must survive verbatim, or models fail to load."""
    text = source_text("lib/shared/segment_cellpose.py")
    for needle in ('model_type="nuclei"', 'size_model_path("nuclei")', 'cellpose_model="nuclei"'):
        assert needle in text, f"Cellpose API string lost: {needle}"


def test_watershed_output_unchanged(synthetic_image):
    """Watershed labels must be byte-identical before and after the rename."""
    from lib.shared.segment_watershed import segment_watershed

    golden = np.load(GOLDEN)
    labels = segment_watershed(
        data=synthetic_image,
        primary_threshold=1000,
        primary_area_min=50,
        primary_area_max=5000,
        cell_threshold=500,
        cells=False,
    )
    np.testing.assert_array_equal(labels, golden)


# Identifiers of ours that must never survive, even on a line that also
# carries Cellpose nomenclature. The line-level check cannot see these.
STALE_IDENTIFIERS = [
    "nuclei_data", "nuclei_diameter", "nuclei_threshold", "nuclei_area_min",
    "nuclei_area_max", "nuclei_flow_threshold", "nuclei_cellprob_threshold",
    "nuclei_prob_threshold", "nuclei_nms_threshold", "nuclei_kwargs",
    "nuclei_model_type", "model_nuclei",
]


@pytest.mark.parametrize("relative_path", SEGMENTATION_LIBS)
def test_no_stale_object_identifiers(relative_path, source_text):
    """Token-level check: catches our identifiers hiding on Cellpose lines."""
    text = source_text(relative_path)
    found = [ident for ident in STALE_IDENTIFIERS if ident in text]
    assert not found, f"{relative_path} still has our old identifiers: {found}"


def test_all_three_methods_importable():
    """Renaming must not break imports for any segmentation method."""
    pytest.importorskip(
        "stardist",
        reason="brieflow_viso pairs numpy 2.0.2 with a stardist built for the numpy 1.x ABI",
        exc_type=ImportError,
    )
    from lib.shared.segment_cellpose import segment_cellpose
    from lib.shared.segment_stardist import segment_stardist
    from lib.shared.segment_watershed import segment_watershed

    for fn in (segment_cellpose, segment_stardist, segment_watershed):
        assert callable(fn)


@pytest.mark.parametrize(
    "module_path,func_name",
    [
        ("lib.shared.segment_cellpose", "segment_cellpose"),
        ("lib.shared.segment_stardist", "segment_stardist"),
        ("lib.shared.segment_watershed", "segment_watershed"),
    ],
)
def test_no_object_name_leaked_into_library_signatures(module_path, func_name):
    """Libraries stay object-agnostic: no 'vacuole' in a shared signature.

    'vacuole' is a display name from config, never a code identifier. Channel
    parameters (dapi_index, cyto_index, threshold_*) must also survive intact.
    """
    import importlib

    if module_path == "lib.shared.segment_stardist":
        pytest.importorskip(
            "stardist",
            reason="brieflow_viso pairs numpy 2.0.2 with a stardist built for the numpy 1.x ABI",
            exc_type=ImportError,
        )

    module = importlib.import_module(module_path)
    params = set(inspect.signature(getattr(module, func_name)).parameters)

    leaked = {p for p in params if "vacuole" in p}
    assert not leaked, f"{func_name} hardcodes a display name: {leaked}"


def test_count_columns_default_to_historical_plural_names():
    """With object_name absent, count columns keep their upstream plural names."""
    from lib.shared.rule_utils import object_plural

    assert object_plural("nucleus") == "nuclei"
    assert object_plural("vacuole") == "vacuoles"

    import pandas as pd

    counts = pd.DataFrame({"final_primary": [1], "initial_primary": [2], "cells": [3]})
    plural = object_plural("nucleus")
    renamed = counts.rename(
        columns={c: c.replace("primary", plural) for c in counts.columns if "primary" in c}
    )
    assert list(renamed.columns) == ["final_nuclei", "initial_nuclei", "cells"]
