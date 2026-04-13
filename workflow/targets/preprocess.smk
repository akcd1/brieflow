from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets
from lib.preprocess.file_utils import get_output_pattern

PREPROCESS_FP = ROOT_FP / "preprocess"

PREPROCESS_OUTPUTS = {
    "extract_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_filename(
            get_output_pattern(phenotype_metadata_wildcard_combos), "metadata", "tsv"
        ),
    ],
    "combine_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "combined_metadata", "parquet"
        ),
    ],
    "convert_phenotype": [
        PREPROCESS_FP / "images" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "image", "tiff"
        ),
    ],
    "calculate_ic_phenotype": [
        PREPROCESS_FP / "ic_fields" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}"},
            "ic_field", "tiff"
        ),
    ],
}

PREPROCESS_OUTPUT_MAPPINGS = {
    "extract_metadata_phenotype": temp,
    "combine_metadata_phenotype": None,
    "convert_phenotype": None,
    "calculate_ic_phenotype": None,
}
PREPROCESS_OUTPUTS_MAPPED = map_outputs(PREPROCESS_OUTPUTS, PREPROCESS_OUTPUT_MAPPINGS)

# Generate targets using the appropriate wildcard combinations
# Metadata extraction uses metadata-specific wildcards
PREPROCESS_TARGETS_PHENOTYPE_METADATA = outputs_to_targets(
    {"extract_metadata_phenotype": PREPROCESS_OUTPUTS["extract_metadata_phenotype"]},
    phenotype_metadata_wildcard_combos,
    PREPROCESS_OUTPUT_MAPPINGS
)

# Other targets use regular wildcard combinations
PREPROCESS_OUTPUTS_PHENOTYPE_OTHER = {
    k: v for k, v in PREPROCESS_OUTPUTS.items()
    if "phenotype" in k and not k.startswith("extract_metadata")
}
PREPROCESS_TARGETS_PHENOTYPE_OTHER = outputs_to_targets(
    PREPROCESS_OUTPUTS_PHENOTYPE_OTHER, phenotype_wildcard_combos, PREPROCESS_OUTPUT_MAPPINGS
)

# Combine all preprocessing targets
PREPROCESS_TARGETS_ALL = (
    PREPROCESS_TARGETS_PHENOTYPE_METADATA + PREPROCESS_TARGETS_PHENOTYPE_OTHER
)
