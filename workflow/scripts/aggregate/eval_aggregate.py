import random

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from lib.aggregate.eval_aggregate import (
    nas_summary,
    plot_feature_distributions,
)

SUBSET_SIZE = 100000

# Early exit if aligned input is empty (e.g. cell_class absent from all wells)
aligned_check = pq.read_table(snakemake.input[0])
if len(aligned_check) == 0:
    print("WARNING: Aligned input is empty. Writing empty outputs and skipping.")
    import pandas as pd
    import matplotlib.pyplot as plt
    pd.DataFrame().to_csv(snakemake.output[0], sep="\t", index=False)
    for out_path in snakemake.output[1:]:
        fig, ax = plt.subplots()
        ax.axis("off")
        fig.savefig(out_path)
        plt.close(fig)
    exit(0)

# Get merge dataset with unified schema to handle null-typed columns
all_schemas = []
for path in snakemake.input.split_datasets_paths:
    try:
        table = pq.read_table(path)
        if len(table) > 0:  # Only use schema from non-empty files
            all_schemas.append(table.schema)
    except Exception as e:
        print(f"Warning: Could not read schema from {path}: {e}")

# Use the first non-empty schema as reference
if all_schemas:
    unified_schema = all_schemas[0]
    merge_data = ds.dataset(snakemake.input.split_datasets_paths, format="parquet", schema=unified_schema)
else:
    print("WARNING: All input files are empty!")
    merge_data = ds.dataset(snakemake.input.split_datasets_paths, format="parquet")

# Choose random row indices
total_rows = merge_data.count_rows()
n_sample = min(SUBSET_SIZE, total_rows)
random_indices = np.random.choice(total_rows, size=n_sample, replace=False)
random_indices.sort()
# Load subset
merge_data = merge_data.scanner().take(random_indices)
merge_data = merge_data.to_pandas(use_threads=True, memory_pool=None).dropna(axis=1)

# Evaluate missing values
nas_df, nas_fig = nas_summary(merge_data, vis_subsample=50000)
nas_df.to_csv(snakemake.output[0], sep="\t", index=False)

# Handle case where no NA values exist (nas_fig will be None)
if nas_fig is not None:
    nas_fig.savefig(snakemake.output[1])
else:
    # Create a placeholder figure when no NA values exist
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(
        0.5, 0.5,
        "No NA values found in dataset\n(All columns with NA values were removed during preprocessing)",
        ha='center', va='center', fontsize=14, wrap=True
    )
    ax.axis('off')
    fig.savefig(snakemake.output[1])
    plt.close(fig)

# Get aligned dataset (single file, so no need for schema unification)
aligned_data = ds.dataset(snakemake.input[0], format="parquet")
# Choose random row indices
total_rows = aligned_data.count_rows()
n_sample = min(SUBSET_SIZE, total_rows)
random_indices = np.random.choice(total_rows, size=n_sample, replace=False)
random_indices.sort()
# Load subset
aligned_data = aligned_data.scanner().take(random_indices)
aligned_data = aligned_data.to_pandas(use_threads=True, memory_pool=None).dropna(axis=1)

# determine original and aligned columns
random.seed(42)
merge_feature_cols = [
    col for col in merge_data.columns if ("cell_" in col and col.endswith("_mean"))
]
pc_cols = [col for col in aligned_data.columns if col.startswith("PC_")]
aligned_feature_cols = random.sample(
    pc_cols, k=min(len(merge_feature_cols), len(pc_cols))
)

# Evaluate feature distributions
feature_distributions_fig = plot_feature_distributions(
    merge_feature_cols,
    merge_data,
    aligned_feature_cols,
    aligned_data,
)
feature_distributions_fig.savefig(snakemake.output[2])
