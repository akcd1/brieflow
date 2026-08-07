import numpy as np

from lib.phenotype.identify_cytoplasm_cellpose import (
    identify_cytoplasm_cellpose,
)
from lib.shared.image_io import read_image, save_image

# Load primary object and cell segmentation data
primary = read_image(snakemake.input[0])
cells = read_image(snakemake.input[1])

# Check if cell segmentation is enabled
segment_cells = snakemake.params.segment_cells

if segment_cells:
    # identify cytoplasms with cellpose
    cytoplasms = identify_cytoplasm_cellpose(primary, cells)
else:
    # write blank array when cell segmentation is disabled
    cytoplasms = np.zeros_like(primary, dtype=np.int32)

# Ensure label array is uint32 (supports >65535 labels; spec-compliant)
cytoplasms = cytoplasms.astype(np.uint32)

# Save cytoplasms data
save_image(cytoplasms, snakemake.output[0], is_label=True)
