from tifffile import imread, imwrite
import pandas as pd
import numpy as np

from lib.phenotype.identify_vacuoles import segment_vacuoles

# Load aligned phenotype image (always present)
data_phenotype = imread(snakemake.input[0])

# Load phenotype info with nuclei centroids (always present)
phenotype_info = pd.read_csv(snakemake.input[-1], sep="\t")

# Determine if cell segmentation is enabled
segment_cells = snakemake.params.get("segment_cells", True)

# Conditionally load cell and cytoplasm masks
if segment_cells and len(snakemake.input) >= 4:
    cells = imread(snakemake.input[1])
    cytoplasms = imread(snakemake.input[2])
    print(f"✓ Cell segmentation enabled: Processing vacuoles with cell association")
else:
    cells = None
    cytoplasms = None
    print(f"✓ Cell segmentation disabled: Processing all vacuoles without cell association")

# Segment vacuoles
result = segment_vacuoles(
    image=data_phenotype,
    vacuole_channel_index=snakemake.params.vacuole_channel_index,
    nuclei_channel_index=snakemake.params.vacuole_channel_index,
    cell_masks=cells,
    cytoplasm_masks=cytoplasms,
    vacuole_min_size=snakemake.params.vacuole_min_size,
    vacuole_max_size=snakemake.params.vacuole_max_size,
    nuclei_centroids=phenotype_info,
    nuclei_detection=snakemake.params.nuclei_detection,
    nuclei_min_distance=snakemake.params.min_distance_between_maxima,
)

# Unpack results based on whether cytoplasm masks were provided
if cytoplasms is not None:
    vacuole_masks, cell_vacuole_table, updated_cytoplasm_masks = result
else:
    vacuole_masks, cell_vacuole_table = result
    # Create empty cytoplasm masks for consistency
    updated_cytoplasm_masks = np.zeros_like(vacuole_masks, dtype=np.uint16)

# Save vacuole masks
imwrite(snakemake.output[0], vacuole_masks)

# Combine and save cell-vacuole tables
cell_summary_df = cell_vacuole_table["cell_summary"]
vacuole_cell_mapping_df = cell_vacuole_table["vacuole_cell_mapping"]

cell_summary_df["table_type"] = "cell_summary"
vacuole_cell_mapping_df["table_type"] = "vacuole_cell_mapping"

# Prefix columns to avoid conflicts
cell_summary_cols = {
    col: f"cell_summary_{col}" 
    for col in cell_summary_df.columns if col != "table_type"
}
vacuole_mapping_cols = {
    col: f"vacuole_mapping_{col}"
    for col in vacuole_cell_mapping_df.columns if col != "table_type"
}

cell_summary_df = cell_summary_df.rename(columns=cell_summary_cols)
vacuole_cell_mapping_df = vacuole_cell_mapping_df.rename(columns=vacuole_mapping_cols)

combined_df = pd.concat([cell_summary_df, vacuole_cell_mapping_df], ignore_index=True)
combined_df.to_csv(snakemake.output[1], sep="\t", index=False)

# Save updated cytoplasm masks
imwrite(snakemake.output[2], updated_cytoplasm_masks)

print(f"✓ Saved vacuole masks, tables, and cytoplasm masks")