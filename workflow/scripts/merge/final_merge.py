import pandas as pd

from lib.shared.file_utils import validate_dtypes

# Load deduplicated merge data
merge_deduplicated = validate_dtypes(pd.read_parquet(snakemake.input[0]))

# Extract configuration parameters
approach = getattr(snakemake.params, "approach", "fast")

# Load full feature data
cp_phenotype = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Load phenotype info (contains alignment metadata)
phenotype_info = validate_dtypes(pd.read_parquet(snakemake.input[2]))

# Merge full CP data on deduplicated
merged_final = merge_deduplicated.merge(
    cp_phenotype.rename(columns={"label": "cell_0"}),
    how="left",
    on=["plate", "well", "tile", "cell_0"],
)

# Merge phenotype_info alignment metadata
# Select only alignment-related columns to avoid duplicating position/bounds columns
alignment_cols = ["plate", "well", "tile", "cell"]
if "FFT_confidence" in phenotype_info.columns:
    alignment_cols.extend(["FFT_confidence", "TM_confidence", "method_used", "offset_x", "offset_y"])

merged_final = merged_final.merge(
    phenotype_info[alignment_cols].rename(columns={"cell": "cell_0"}),
    how="left",
    on=["plate", "well", "tile", "cell_0"],
)

# Rename coordinate columns to global naming convention only for stitch approach
if snakemake.params.approach == "stitch":
    merged_final = merged_final.rename(
        columns={
            "i_0": "global_i_0",
            "j_0": "global_j_0",
            "i_1": "global_i_1",
            "j_1": "global_j_1",
        }
    )

# Save final merged dataset
merged_final.to_parquet(snakemake.output[0])
