import pandas as pd
from itertools import product

from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


CLUSTER_FP = ROOT_FP / "cluster"

CLUSTER_OUTPUTS = {
    "clean_aggregate": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / get_filename({}, "aggregate_cleaned", "tsv"),
    ],
    # Higher-dimensional (50-D) PHATE embedding for point-to-point distance
    # measurement in the same diffusion space the clustering uses (identical PHATE
    # settings, just 50 output coordinates instead of 2). JOINT only (the joint
    # aggregate is the shared clustering space) and independent of leiden_resolution.
    # Row order matches the row-ids file (class:cell_barcode_0).
    "phate_diff_embedding": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "joint"
        / get_filename({}, "phate_int50", "npy"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "joint"
        / get_filename({}, "phate_int50_row_ids", "txt"),
    ],
    "phate_leiden_clustering": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {},
            "phate_leiden_clustering",
            "tsv",
        ),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({}, "cluster_sizes", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({}, "clusters", "png"),
    ],
    "benchmark_clusters": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "integrated_results", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "integrated_results", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "combined_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "combined_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "global_metrics", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "global_metrics", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "pie_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {"cluster_benchmark": "Shuffled"}, "enrichment_pie_chart", "png"
        ),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "enrichment_bar_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{compartment_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {"cluster_benchmark": "Shuffled"}, "enrichment_bar_chart", "png"
        ),
    ],
}

CLUSTER_OUTPUT_MAPPINGS = {
    "clean_aggregate": None,
    "phate_diff_embedding": None,
    "phate_leiden_clustering": None,
    "benchmark_clusters": None,
}


# TODO: Use all combos
# cluster_wildcard_combos = cluster_wildcard_combos[
#     (cluster_wildcard_combos["cell_class"].isin(["Interphase"]))
#     & (cluster_wildcard_combos["channel_combo"].isin(["DAPI_COXIV_CENPA_WGA"]))
#     & (cluster_wildcard_combos["leiden_resolution"].isin([13]))
# ]

# Determine which outputs to include based on config. Default True for
# backwards compatibility — existing configs without the flag keep benchmarking.
CLUSTER_RUN_BENCHMARK = config["cluster"].get("run_benchmark", True)

if not CLUSTER_RUN_BENCHMARK:
    CLUSTER_OUTPUTS_FILTERED = {
        k: v for k, v in CLUSTER_OUTPUTS.items() if k != "benchmark_clusters"
    }
    CLUSTER_OUTPUT_MAPPINGS_FILTERED = {
        k: v for k, v in CLUSTER_OUTPUT_MAPPINGS.items() if k != "benchmark_clusters"
    }
else:
    CLUSTER_OUTPUTS_FILTERED = CLUSTER_OUTPUTS
    CLUSTER_OUTPUT_MAPPINGS_FILTERED = CLUSTER_OUTPUT_MAPPINGS

CLUSTER_OUTPUTS_MAPPED = map_outputs(
    CLUSTER_OUTPUTS_FILTERED, CLUSTER_OUTPUT_MAPPINGS_FILTERED
)

CLUSTER_TARGETS_ALL = outputs_to_targets(
    CLUSTER_OUTPUTS_FILTERED, cluster_wildcard_combos, CLUSTER_OUTPUT_MAPPINGS_FILTERED
)
