from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


PHENOTYPE_FP = ROOT_FP / "phenotype"

# determine feature eval outputs based on channel names
channel_names = config["phenotype"]["channel_names"]
eval_features = [f"cell_{channel}_min" for channel in channel_names]

PHENOTYPE_OUTPUTS = {
    "apply_ic_field_phenotype": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "illumination_corrected",
            "tiff",
        ),
    ],
    "align_phenotype": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "aligned", "tiff"
        ),
    ],
    "segment_phenotype": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "nuclei", "tiff"
        ),
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "cells", "tiff"
        ),
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "segmentation_stats",
            "tsv",
        ),
    ],
    "identify_cytoplasm": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "identified_cytoplasms",
            "tiff",
        ),
    ],
    "extract_phenotype_info": [
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "phenotype_info",
            "tsv",
        ),
    ],
    "combine_phenotype_info": [
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_info", "parquet"
        ),
    ],
    "identify_vacuoles": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "identified_vacuoles",
            "tiff",
        ),
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "cell_vacuole_table",
            "tsv",
        ),
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "updated_cytoplasms",
            "tiff",
        ),
    ],
    "extract_phenotype_cp": [
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "phenotype_cp",
            "tsv",
        ),
    ],
    "extract_phenotype_vacuoles": [
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "phenotype_vacuoles",
            "tsv",
        ),
    ],
    "merge_phenotype_vacuoles": [
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_vacuoles", "parquet"
        ),
    ],
    "merge_vacuoles_phenotype_cp": [
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "phenotype_with_vacuoles",
            "tsv",
        ),
    ],
    "merge_phenotype_cp": [
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_cp", "parquet"
        ),
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_cp_min", "parquet"
        ),
    ],
    "eval_segmentation_phenotype": [
        PHENOTYPE_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "segmentation_overview", "tsv"),
        PHENOTYPE_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "cell_density_heatmap", "tsv"),
        PHENOTYPE_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "cell_density_heatmap", "png"),
    ],
    # create heatmap tsv and png for each evaluated feature
    "eval_features": [
        PHENOTYPE_FP
        / "eval"
        / "features"
        / get_filename({"plate": "{plate}"}, f"{feature}_heatmap", "tsv")
        for feature in eval_features
    ]
    + [
        PHENOTYPE_FP
        / "eval"
        / "features"
        / get_filename({"plate": "{plate}"}, f"{feature}_heatmap", "png")
        for feature in eval_features
    ],
}

PHENOTYPE_OUTPUT_MAPPINGS = {
    "apply_ic_field_phenotype": temp,
    "align_phenotype": None,
    "segment_phenotype": None,
    "identify_cytoplasm": temp,
    "extract_phenotype_info": temp,
    "combine_phenotype_info": None,
    "identify_vacuoles": None,
    "extract_phenotype_cp": temp,
    "extract_phenotype_vacuoles": temp,
    "merge_phenotype_vacuoles": None,
    "merge_vacuoles_phenotype_cp": None,
    "merge_phenotype_cp": None,
    "eval_segmentation_phenotype": None,
    "eval_features": None,
}

# TODO: test and implement segmentation paramsearch for updated brieflow setup
# if config["phenotype"]["mode"] == "segment_phenotype_paramsearch":
#     PHENOTYPE_OUTPUTS.update(
#         {
#             "segment_phenotype_paramsearch": [
#                 PHENOTYPE_FP
#                 / "paramsearch"
#                 / "images"
#                 / get_filename(
#                     {"well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_nuclei",
#                     "tiff",
#                 ),
#                 PHENOTYPE_FP
#                 / "paramsearch"
#                 / "images"
#                 / get_filename(
#                     {"well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_cells",
#                     "tiff",
#                 ),
#                 PHENOTYPE_FP
#                 / "paramsearch"
#                 / "tsvs"
#                 / get_filename(
#                     {"well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_segmentation_stats",
#                     "tsv",
#                 ),
#             ],
#             "summarize_segment_phenotype_paramsearch": [
#                 PHENOTYPE_FP / "paramsearch" / "summary" / "segmentation_summary.tsv",
#                 PHENOTYPE_FP / "paramsearch" / "summary" / "segmentation_grouped.tsv",
#                 PHENOTYPE_FP
#                 / "paramsearch"
#                 / "summary"
#                 / "segmentation_evaluation.txt",
#                 PHENOTYPE_FP / "paramsearch" / "summary" / "segmentation_panel.png",
#             ],
#         }
#     )

#     PHENOTYPE_OUTPUT_MAPPINGS.update(
#         {
#             "segment_phenotype_paramsearch": None,
#             "summarize_segment_phenotype_paramsearch": None,
#         }
#     )

#     PHENOTYPE_WILDCARDS.update(
#         {
#             "nuclei_diameter": config["phenotype"]["paramsearch"]["nuclei_diameter"],
#             "cell_diameter": config["phenotype"]["paramsearch"]["cell_diameter"],
#             "flow_threshold": config["phenotype"]["paramsearch"]["flow_threshold"],
#             "cellprob_threshold": config["phenotype"]["paramsearch"][
#                 "cellprob_threshold"
#             ],
#         }
#     )


PHENOTYPE_OUTPUTS_MAPPED = map_outputs(PHENOTYPE_OUTPUTS, PHENOTYPE_OUTPUT_MAPPINGS)

# =====================================================================
# MODIFIED: Conditional targets based on segment_cells parameter
# =====================================================================

# Define base outputs that are ALWAYS required (work with or without cells)
ALWAYS_REQUIRED_OUTPUTS = [
    "apply_ic_field_phenotype",
    "align_phenotype",
    "segment_phenotype",
    "extract_phenotype_info",
    "combine_phenotype_info",
    "identify_vacuoles",
    "extract_phenotype_vacuoles",
    "merge_phenotype_vacuoles",
]

# Define outputs that are ONLY required when segment_cells=true
CELL_DEPENDENT_OUTPUTS = [
    "identify_cytoplasm",
    "extract_phenotype_cp",
    "merge_vacuoles_phenotype_cp",
    "merge_phenotype_cp",
    "eval_segmentation_phenotype",
    "eval_features",
]

# Build the targets list conditionally
if config["phenotype"]["segment_cells"]:
    # WITH CELL SEGMENTATION: Include all outputs
    outputs_to_include = ALWAYS_REQUIRED_OUTPUTS + CELL_DEPENDENT_OUTPUTS
    print("✓ Phenotype workflow: Cell segmentation enabled - including all outputs")
else:
    # WITHOUT CELL SEGMENTATION: Only include base outputs
    outputs_to_include = ALWAYS_REQUIRED_OUTPUTS
    print("✓ Phenotype workflow: Cell segmentation disabled - skipping cell-dependent outputs")

# Filter PHENOTYPE_OUTPUTS to only include the desired outputs
PHENOTYPE_OUTPUTS_FILTERED = {
    key: value for key, value in PHENOTYPE_OUTPUTS.items() 
    if key in outputs_to_include
}

# Filter PHENOTYPE_OUTPUT_MAPPINGS to match
PHENOTYPE_OUTPUT_MAPPINGS_FILTERED = {
    key: value for key, value in PHENOTYPE_OUTPUT_MAPPINGS.items() 
    if key in outputs_to_include
}

# Generate targets from the filtered outputs
PHENOTYPE_TARGETS_ALL = outputs_to_targets(
    PHENOTYPE_OUTPUTS_FILTERED, 
    phenotype_wildcard_combos, 
    PHENOTYPE_OUTPUT_MAPPINGS_FILTERED
)

# Debug information (optional - uncomment to see what's happening)
# print(f"  - Total output types: {len(PHENOTYPE_OUTPUTS_FILTERED)}/{len(PHENOTYPE_OUTPUTS)}")
# print(f"  - Total target files: {len(PHENOTYPE_TARGETS_ALL)}")
