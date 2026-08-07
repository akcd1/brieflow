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


def test_metadata_cols_default_is_backward_compatible():
    """Absent object_name must reproduce the existing column list exactly."""
    from lib.phenotype.constants import DEFAULT_METADATA_COLS, metadata_cols

    assert metadata_cols() == DEFAULT_METADATA_COLS


def test_metadata_cols_renames_only_the_object_columns():
    """object_name drives the object columns and nothing else."""
    from lib.phenotype.constants import metadata_cols

    cols = metadata_cols("vacuole")
    for expected in ("vacuole_i", "vacuole_j", "vacuole_bounds_0", "vacuole_bounds_3"):
        assert expected in cols, f"missing {expected}"
    assert not any(c.startswith("nucleus_") for c in cols)
    # Cell and cytoplasm columns are a different object; they must be untouched.
    assert "cell_i" in cols
    assert "cytoplasm_i" in cols


def test_feature_prefix_is_not_hardcoded(source_text):
    """The prefix must come from object_name, never a literal."""
    emulator = source_text("lib/phenotype/extract_phenotype_cp_emulator.py")
    assert 'add_prefix("nucleus_")' not in emulator, "prefix still hardcoded"
    assert 'add_prefix(f"{object_name}_")' in emulator


# A signature default of the form `object_name: str = "nucleus"` is the
# legitimate, backward-compatible fallback established by Task 6 -- it is
# not a hardcoded *usage* of the object name. Everything else quoting
# "nucleus" or "nucleus_" in these files is a regression: it means some code
# path selects/labels columns without going through object_name, which is
# exactly the bug that crashed merge_phenotype.py under harissa's config.
ALLOWED_NUCLEUS_DEFAULT = 'object_name: str = "nucleus"'


def test_no_hardcoded_object_prefix_in_phenotype_consumers(source_text):
    """Every consumer of the prefixed columns must derive the name, not hardcode it."""
    for rel in [
        "lib/phenotype/extract_phenotype_cp_emulator.py",
        "lib/phenotype/extract_phenotype_cp_measure.py",
        "scripts/phenotype/merge_phenotype.py",
    ]:
        text = source_text(rel)
        offenders = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ALLOWED_NUCLEUS_DEFAULT in line:
                continue
            if '"nucleus_"' in line or '"nucleus"' in line:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert not offenders, f"hardcoded object prefix/name survived: {offenders}"


def test_label_basename_derives_from_config():
    """The segment_phenotype label output must derive its name, not hardcode it."""
    import pathlib

    targets = (
        pathlib.Path(__file__).resolve().parents[2] / "workflow" / "targets" / "phenotype.smk"
    ).read_text()

    assert "object_plural(object_name)" in targets, (
        "label basename no longer derives from object_name"
    )
    assert '"nuclei"' not in targets, "label basename is hardcoded again"


def test_object_plural_handles_the_irregular_case():
    """nucleus -> nuclei, not nucleuss; the reason the helper exists."""
    from lib.shared.rule_utils import object_plural

    assert object_plural("nucleus") == "nuclei"
    assert object_plural("vacuole") == "vacuoles"
