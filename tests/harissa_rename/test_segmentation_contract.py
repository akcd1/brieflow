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


# ---------------------------------------------------------------------------
# Regression tests: independent review CRITICAL 1 & CRITICAL 2
#
# CRITICAL 1: get_segmentation_params() read only the post-rename primary_*
# keys. A config still written with the pre-rename nuclei_* keys silently
# got None / a hardcoded default back from .get() -- segmentation params
# differed from what the user configured, with no error.
#
# CRITICAL 2: the OME-Zarr segmentation_metadata map was keyed on the
# literal "nuclei", so with object_name="vacuole" the real primary label
# store (vacuoles.zarr) got no metadata at all, while all-zero placeholder
# "cells"/"identified_cytoplasms" stores (written when segment_cells=False)
# got metadata claiming a cell segmentation that never ran.
# ---------------------------------------------------------------------------


def test_legacy_diameter_key_still_resolves_with_warning():
    """A pre-rename config (nuclei_diameter) must still work, loudly."""
    import warnings

    from lib.shared.rule_utils import get_segmentation_params

    legacy_config = {
        "phenotype": {
            "segmentation_method": "cellpose",
            "object_name": "vacuole",
            "nuclei_diameter": 41.97813156768016,
            "nuclei_flow_threshold": 0.4,
            "nuclei_cellprob_threshold": 0.0,
        }
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = get_segmentation_params("phenotype", legacy_config)

    assert params["primary_diameter"] == 41.97813156768016
    assert params["primary_flow_threshold"] == 0.4
    assert params["primary_cellprob_threshold"] == 0.0

    messages = [str(w.message) for w in caught]
    assert any(
        "nuclei_diameter" in m and "primary_diameter" in m for m in messages
    ), f"no warning naming both keys was emitted: {messages}"


def test_bundled_config_primary_diameter_resolves_not_none():
    """The repo's own bundled test config still uses nuclei_diameter (line
    194 of tests/small_test_analysis/config/config.yml). Before the fix,
    get_segmentation_params returned primary_diameter=None here and Cellpose
    silently auto-sized instead of using the configured 41.978um diameter.
    """
    from pathlib import Path

    import yaml

    from lib.shared.rule_utils import get_segmentation_params

    config_fp = (
        Path(__file__).resolve().parents[1]
        / "small_test_analysis"
        / "config"
        / "config.yml"
    )
    with open(config_fp) as f:
        config = yaml.safe_load(f)

    params = get_segmentation_params("phenotype", config)
    assert params["primary_diameter"] == pytest.approx(41.97813156768016)


def test_watershed_area_bounds_use_legacy_key_not_silent_default():
    """Watershed's primary_area_min/max must fall back to nuclei_area_min/max,
    not silently drop to the 45/450 hardcoded defaults.
    """
    from lib.shared.rule_utils import get_segmentation_params

    config = {
        "phenotype": {
            "segmentation_method": "watershed",
            "nuclei_area_min": 12,
            "nuclei_area_max": 999,
        }
    }
    params = get_segmentation_params("phenotype", config)
    assert params["primary_area_min"] == 12
    assert params["primary_area_max"] == 999


def test_primary_store_recognized_under_nondefault_object_name():
    """With object_name='vacuole', the primary label store is 'vacuoles.zarr'
    (stem 'vacuoles'), never the literal 'nuclei'.
    """
    pytest.importorskip(
        "iohub",
        reason="iohub not installed in the borrowed brieflow_viso test env; "
        "lib.shared.hcs needs it for OME-Zarr metadata patching",
        exc_type=ModuleNotFoundError,
    )
    from lib.shared.hcs import _build_segmentation_meta_for_label

    modality_config = {
        "object_name": "vacuole",
        "segmentation_method": "cellpose",
        "cellpose_model": "cyto3",
        "dapi_index": 0,
        "primary_diameter": 41.978,
    }

    # The literal "nuclei" must NOT match once object_name is renamed --
    # this is the object that was never segmented under this config.
    assert _build_segmentation_meta_for_label("nuclei", modality_config, None) is None

    meta = _build_segmentation_meta_for_label("vacuoles", modality_config, None)
    assert meta is not None
    assert meta["annotation_type"] == "vacuole"
    assert meta["segmentation"]["parameters"]["primary_diameter"] == 41.978


def test_default_object_name_diameter_parameters_populated():
    """Even the unrenamed default case was broken: diameter_key pointed at
    the pre-rename config key ('nuclei_diameter'), so `parameters` came back
    empty. Prove it is now populated for the default object_name too.
    """
    pytest.importorskip(
        "iohub",
        reason="iohub not installed in the borrowed brieflow_viso test env; "
        "lib.shared.hcs needs it for OME-Zarr metadata patching",
        exc_type=ModuleNotFoundError,
    )
    from lib.shared.hcs import _build_segmentation_meta_for_label

    modality_config = {
        "segmentation_method": "cellpose",
        "cellpose_model": "cyto3",
        "dapi_index": 0,
        "primary_diameter": 9.5,
    }
    meta = _build_segmentation_meta_for_label("nuclei", modality_config, None)
    assert meta["segmentation"]["parameters"]["primary_diameter"] == 9.5


def test_hcs_metadata_resolves_legacy_diameter_key_too():
    """hcs.py receives the raw config section (not run through
    get_segmentation_params), so it needs the same nuclei_* legacy fallback
    independently.
    """
    pytest.importorskip(
        "iohub",
        reason="iohub not installed in the borrowed brieflow_viso test env; "
        "lib.shared.hcs needs it for OME-Zarr metadata patching",
        exc_type=ModuleNotFoundError,
    )
    from lib.shared.hcs import _build_segmentation_meta_for_label

    modality_config = {
        "segmentation_method": "cellpose",
        "cellpose_model": "cyto3",
        "dapi_index": 0,
        "nuclei_diameter": 41.97813156768016,
    }
    meta = _build_segmentation_meta_for_label("nuclei", modality_config, None)
    assert (
        meta["segmentation"]["parameters"]["primary_diameter"]
        == 41.97813156768016
    )


def test_placeholder_allzero_label_gets_no_segmentation_metadata(tmp_path):
    """cells/identified_cytoplasms are written as all-zero placeholder
    arrays when segment_cells=False. They must not be tagged with metadata
    claiming a real segmentation happened.
    """
    pytest.importorskip(
        "iohub",
        reason="iohub not installed in the borrowed brieflow_viso test env; "
        "lib.shared.hcs needs it for OME-Zarr metadata patching",
        exc_type=ModuleNotFoundError,
    )
    import json

    import zarr

    from lib.shared.hcs import _patch_segmentation_metadata

    store_path = tmp_path / "aligned_1.zarr"
    label_dir = store_path / "A" / "1" / "0" / "labels" / "cells.zarr"
    arr_dir = label_dir / "0"
    label_dir.mkdir(parents=True, exist_ok=True)

    (label_dir / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": {"image-label": {"version": "0.5"}}},
            }
        )
    )

    z = zarr.open(str(arr_dir), mode="w", shape=(8, 8), dtype="uint16")
    z[:] = 0

    modality_config = {"segmentation_method": "cellpose", "segment_cells": False}
    _patch_segmentation_metadata(store_path, modality_config, None)

    meta = json.loads((label_dir / "zarr.json").read_text())
    assert "segmentation_metadata" not in meta.get("attributes", {})


# ---------------------------------------------------------------------------
# Regression test: independent review IMPORTANT 2
#
# object_name="cell" (or "cells"/"cytoplasm"/"cytoplasms") collides with
# names hardcoded elsewhere in the phenotype pipeline: with object_name=
# "cell", hcs._label_annotation_map collapses from 3 keys to 2 (the primary
# entry is silently overwritten by the literal "cells" entry), and
# order_dataframe_columns duplicates columns that match both the
# f"{object_name}_" and "cell_" prefixes. Nothing rejected it. get_segmentation_params
# must now fail loudly at config-read time instead.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reserved_name", ["cell", "cells", "cytoplasm", "cytoplasms"])
def test_reserved_object_name_rejected(reserved_name):
    """Reserved object_name values must raise ValueError, not corrupt output."""
    from lib.shared.rule_utils import get_segmentation_params

    config = {
        "phenotype": {
            "segmentation_method": "cellpose",
            "object_name": reserved_name,
        }
    }
    with pytest.raises(ValueError, match=reserved_name):
        get_segmentation_params("phenotype", config)


def test_non_reserved_object_name_still_accepted():
    """The reserved-name check must not reject legitimate object names."""
    from lib.shared.rule_utils import get_segmentation_params

    config = {
        "phenotype": {
            "segmentation_method": "cellpose",
            "object_name": "vacuole",
        }
    }
    params = get_segmentation_params("phenotype", config)
    assert params["object_name"] == "vacuole"


def test_real_segmentation_label_still_gets_metadata(tmp_path):
    """Sanity check: a label store with real (non-zero) objects still gets
    segmentation_metadata written, so the all-zero skip isn't overbroad.
    """
    pytest.importorskip(
        "iohub",
        reason="iohub not installed in the borrowed brieflow_viso test env; "
        "lib.shared.hcs needs it for OME-Zarr metadata patching",
        exc_type=ModuleNotFoundError,
    )
    import json

    import zarr

    from lib.shared.hcs import _patch_segmentation_metadata

    store_path = tmp_path / "aligned_1.zarr"
    label_dir = store_path / "A" / "1" / "0" / "labels" / "vacuoles.zarr"
    arr_dir = label_dir / "0"
    label_dir.mkdir(parents=True, exist_ok=True)

    (label_dir / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": {"image-label": {"version": "0.5"}}},
            }
        )
    )

    z = zarr.open(str(arr_dir), mode="w", shape=(4, 4), dtype="uint16")
    z[0:2, 0:2] = 1
    z[2:4, 2:4] = 2

    modality_config = {
        "object_name": "vacuole",
        "segmentation_method": "cellpose",
        "cellpose_model": "cyto3",
        "dapi_index": 0,
        "primary_diameter": 41.978,
    }
    _patch_segmentation_metadata(store_path, modality_config, None)

    meta = json.loads((label_dir / "zarr.json").read_text())
    seg_meta = meta["attributes"]["segmentation_metadata"]
    assert seg_meta["annotation_type"] == "vacuole"
    assert seg_meta["statistics"]["n_cells"] == 2
