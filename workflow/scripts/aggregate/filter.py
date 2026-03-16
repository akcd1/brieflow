import pandas as pd

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.filter import (
    query_filter,
    perturbation_filter,
    missing_values_filter,
    intensity_filter,
)

# Load cell data
cell_data = pd.read_parquet(snakemake.input[0])
use_classifier = snakemake.params.get("use_classifier", False)
metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp,
    include_classification_cols=use_classifier,
)
metadata, features = split_cell_data(cell_data, metadata_cols)

# Early exit if input is empty (e.g., cell class not present in this well)
if len(metadata) == 0:
    print("WARNING: Input data is empty. Creating empty output file.")
    cell_data = pd.concat([metadata, features], axis=1)
    cell_data.to_parquet(snakemake.output[0], index=False)
    exit(0)

# Filter
metadata, features = query_filter(
    metadata,
    features,
    snakemake.params.filter_queries,
)
metadata, features = perturbation_filter(
    metadata,
    features,
    snakemake.params.perturbation_name_col,
)

# Early exit if no cells remain after perturbation filtering
if len(metadata) == 0:
    print("WARNING: No cells with perturbations after filtering. Creating empty output file.")
    cell_data = pd.concat([metadata, features], axis=1)
    cell_data.to_parquet(snakemake.output[0], index=False)
    exit(0)

metadata, features = missing_values_filter(
    metadata,
    features,
    drop_cols_threshold=snakemake.params.drop_cols_threshold,
    drop_rows_threshold=snakemake.params.drop_rows_threshold,
    impute=snakemake.params.impute,
)
metadata, features = intensity_filter(
    metadata,
    features,
    snakemake.params.channel_names,
    snakemake.params.contamination,
)

# Save filtered data
cell_data = pd.concat([metadata, features], axis=1)
cell_data.to_parquet(snakemake.output[0], index=False)
