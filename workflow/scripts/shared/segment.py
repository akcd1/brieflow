import numpy as np
import pandas as pd

from lib.shared.image_io import read_image, save_image

# Load illumination corrected data
aligned_data = read_image(snakemake.input[0])

# Get configuration from params
params = snakemake.params.config

# Choose segmentation method based on parameter
method = params.get("segmentation_method", "cellpose")
segment_cells = params.get("segment_cells", True)

if method == "cellpose":
    # Segment cells using cellpose
    from lib.shared.segment_cellpose import segment_cellpose

    if segment_cells:
        primary_data, cells_data, counts_df = segment_cellpose(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            primary_diameter=params["primary_diameter"],
            cell_diameter=params["cell_diameter"],
            cellpose_model=params["cellpose_model"],
            helper_index=params.get("helper_index"),
            cellpose_kwargs=dict(
                flow_threshold=params.get("flow_threshold", 0.4),
                cellprob_threshold=params.get("cellprob_threshold", 0),
                primary_flow_threshold=params["primary_flow_threshold"],
                primary_cellprob_threshold=params["primary_cellprob_threshold"],
                cell_flow_threshold=params["cell_flow_threshold"],
                cell_cellprob_threshold=params["cell_cellprob_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
    else:
        primary_data, counts_df = segment_cellpose(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            primary_diameter=params["primary_diameter"],
            cell_diameter=params["cell_diameter"],
            cellpose_model=params["cellpose_model"],
            helper_index=params.get("helper_index"),
            cellpose_kwargs=dict(
                flow_threshold=params.get("flow_threshold", 0.4),
                cellprob_threshold=params.get("cellprob_threshold", 0),
                primary_flow_threshold=params["primary_flow_threshold"],
                primary_cellprob_threshold=params["primary_cellprob_threshold"],
                cell_flow_threshold=params["cell_flow_threshold"],
                cell_cellprob_threshold=params["cell_cellprob_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
        cells_data = np.zeros_like(primary_data)

elif method == "stardist":
    # Segment cells using StarDist
    from lib.shared.segment_stardist import segment_stardist

    if segment_cells:
        primary_data, cells_data, counts_df = segment_stardist(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["stardist_model"],
            stardist_kwargs=dict(
                prob_threshold=params.get("prob_threshold", 0.479071),
                nms_threshold=params.get("nms_threshold", 0.3),
                primary_prob_threshold=params["primary_prob_threshold"],
                primary_nms_threshold=params["primary_nms_threshold"],
                cell_prob_threshold=params["cell_prob_threshold"],
                cell_nms_threshold=params["cell_nms_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
    else:
        primary_data, counts_df = segment_stardist(
            data=aligned_data,
            dapi_index=params["dapi_index"],
            cyto_index=params["cyto_index"],
            model_type=params["stardist_model"],
            stardist_kwargs=dict(
                prob_threshold=params.get("prob_threshold", 0.479071),
                nms_threshold=params.get("nms_threshold", 0.3),
                primary_prob_threshold=params["primary_prob_threshold"],
                primary_nms_threshold=params["primary_nms_threshold"],
                cell_prob_threshold=params["cell_prob_threshold"],
                cell_nms_threshold=params["cell_nms_threshold"],
            ),
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
            gpu=params.get("gpu", False),
            cells=segment_cells,
        )
        cells_data = np.zeros_like(primary_data)

elif method == "watershed":
    # Segment cells using Watershed
    from lib.shared.segment_watershed import segment_watershed

    if segment_cells:
        primary_data, cells_data, counts_df = segment_watershed(
            data=aligned_data,
            primary_threshold=params["threshold_dapi"],
            primary_area_min=params["primary_area_min"],
            primary_area_max=params["primary_area_max"],
            cell_threshold=params["threshold_cell"],
            cells=True,
            reconcile=params.get("reconcile"),
            return_counts=params.get("return_counts", True),
        )
    else:
        primary_data, counts_df = segment_watershed(
            data=aligned_data,
            primary_threshold=params["threshold_dapi"],
            primary_area_min=params["primary_area_min"],
            primary_area_max=params["primary_area_max"],
            cell_threshold=params["threshold_cell"],
            return_counts=params.get("return_counts", True),
            cells=segment_cells,
        )
        cells_data = np.zeros_like(primary_data)
else:
    raise ValueError(
        f"Unknown segmentation method: {method}. Choose one of: cellpose, stardist, watershed"
    )

# Ensure label arrays are uint32 (supports >65535 labels; spec-compliant)
primary_data = primary_data.astype(np.uint32)
cells_data = cells_data.astype(np.uint32)

# Save segmented primary object data
save_image(primary_data, snakemake.output[0], is_label=True)
# Save segmented cells data
save_image(cells_data, snakemake.output[1], is_label=True)

# The libraries emit neutral "*primary*" keys; the stats file is user-facing,
# so rename them to whatever this screen calls its object.
object_name = params.get("object_name", "nucleus")
counts_df = counts_df.rename(
    columns={col: col.replace("primary", object_name) for col in counts_df.columns if "primary" in col}
)

# Save counts data
counts_df.to_csv(snakemake.output[2], index=False, sep="\t")
