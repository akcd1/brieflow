import pandas as pd

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.filter import (
    query_filter,
    perturbation_filter,
    missing_values_filter,
    intensity_filter,
)


def write_empty_and_exit(metadata, features, output_path, stage):
    """Write empty parquet and exit if no cells remain after a filtering stage."""
    if len(metadata) == 0:
        print(f"WARNING: No cells after {stage}, writing empty output")
        pd.concat([metadata, features], axis=1).to_parquet(output_path, index=False)
        exit(0)


output_path = snakemake.output[0]
use_classifier = snakemake.params.get("use_classifier", False)
metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp,
    include_classification_cols=use_classifier,
)

# Global and per-class query config
global_queries = list(snakemake.params.filter_queries or [])
per_class_queries: dict = snakemake.params.filter_queries_by_class or {}

is_joint = snakemake.wildcards.cell_class == "joint"

if is_joint:
    # --- JOINT MODE: per-class query+pert filter, then pool, then global filters ---
    input_paths = list(snakemake.input)
    print(f"JOINT filter: pooling {len(input_paths)} per-class inputs")

    pooled_parts = []
    for path in input_paths:
        part = pd.read_parquet(path)
        if len(part) == 0:
            continue
        part_metadata, part_features = split_cell_data(part, metadata_cols)

        # Use the 'class' column placed by split_datasets; each per-class parquet
        # has a single class value, but read from the column (not the wildcard).
        part_class = (
            part_metadata["class"].iloc[0] if "class" in part_metadata.columns else None
        )

        # Apply global queries + this class's per-class queries
        queries = global_queries + list(per_class_queries.get(part_class, []))
        part_metadata, part_features = query_filter(
            part_metadata, part_features, queries
        )
        if len(part_metadata) == 0:
            continue
        part_metadata, part_features = perturbation_filter(
            part_metadata, part_features, snakemake.params.perturbation_name_col
        )
        if len(part_metadata) == 0:
            continue

        pooled_parts.append(pd.concat([part_metadata, part_features], axis=1))

    if not pooled_parts:
        print(
            "WARNING: All joint inputs empty after per-class filters; writing empty output"
        )
        pd.DataFrame().to_parquet(output_path, index=False)
        exit(0)

    pooled = pd.concat(pooled_parts, axis=0, ignore_index=True)
    metadata, features = split_cell_data(pooled, metadata_cols)

    # Assertion: feature column set is identical across class slices
    class_values = metadata["class"].unique()
    feature_cols_sets = {
        cls: set(features.loc[metadata["class"] == cls].columns) for cls in class_values
    }
    reference = feature_cols_sets[class_values[0]]
    for cls, cols in feature_cols_sets.items():
        assert cols == reference, (
            f"Feature columns diverge between classes '{class_values[0]}' and '{cls}' "
            f"before global filtering. This should not happen with approach C."
        )
    print(
        f"JOINT filter: pooled {len(metadata)} cells across classes {list(class_values)}"
    )

else:
    # --- NON-JOINT MODE: existing single-input behavior ---
    input_path = (
        snakemake.input[0] if not isinstance(snakemake.input, str) else snakemake.input
    )
    cell_data = pd.read_parquet(input_path)
    metadata, features = split_cell_data(cell_data, metadata_cols)

    # Apply global + this wildcard's per-class queries
    queries = global_queries + list(
        per_class_queries.get(snakemake.wildcards.cell_class, [])
    )
    metadata, features = query_filter(metadata, features, queries)
    write_empty_and_exit(metadata, features, output_path, "query_filter")

    metadata, features = perturbation_filter(
        metadata,
        features,
        snakemake.params.perturbation_name_col,
    )
    write_empty_and_exit(metadata, features, output_path, "perturbation_filter")

# --- GLOBAL FILTERS (both modes) ---
metadata, features = missing_values_filter(
    metadata,
    features,
    drop_cols_threshold=snakemake.params.drop_cols_threshold,
    drop_rows_threshold=snakemake.params.drop_rows_threshold,
    impute=snakemake.params.impute,
)
write_empty_and_exit(metadata, features, output_path, "missing_values_filter")

metadata, features = intensity_filter(
    metadata,
    features,
    snakemake.params.channel_names,
    snakemake.params.contamination,
)

cell_data = pd.concat([metadata, features], axis=1)
cell_data.to_parquet(output_path, index=False)
