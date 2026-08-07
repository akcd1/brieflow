"""Cellpose-based Image Segmentation!

This module provides functions for segmenting microscopy images using the Cellpose algorithm
(relating to SBS base calling and phenotyping -- steps 1 and 2). It includes functions for:

1. Cell and Primary Segmentation: Segmenting cells and primary from various image types.
2. Image Preprocessing: Applying log scaling and other preprocessing techniques to images.
3. Label Reconciliation: Reconciling primary and cell labels based on their spatial relationships.
4. Mask Processing: Manipulating and refining segmentation masks.
5. Utility Functions: Supporting operations for image analysis and segmentation tasks.

COMPATIBILITY NOTE:
This module supports both Cellpose 3.x and 4.x with automatic version detection:

Cellpose 3.x (3.1.0):
- Supports model_type values: cyto3, nuclei, cyto2
- Supports automatic diameter estimation
- CPSAM model NOT supported

Cellpose 4.x (4.0.4+):
- Supports ONLY cpsam model
- Standard model_type values (cyto3, nuclei) NOT supported in 4.x
- Automatic diameter estimation NOT supported (specify diameters in config)
- Upgrade: uv pip install cellpose==4.0.4 torch==2.7.0 torchvision==0.22.0

The code will automatically detect your Cellpose version and raise clear errors
for incompatible model/version combinations.

"""

import sys

import numpy as np
import pandas as pd

import cellpose
from cellpose.models import CellposeModel
from cellpose import models as cellpose_models
from skimage.util import img_as_ubyte
from skimage.segmentation import clear_border

from lib.shared.segmentation_utils import (
    image_log_scale,
    reconcile_primary_cells,
)

# Detect Cellpose version for compatibility checks
try:
    CELLPOSE_VERSION = tuple(map(int, cellpose.version.split(".")[:2]))
except (AttributeError, ValueError):
    # Fallback to safe default for backward compatibility
    CELLPOSE_VERSION = (3, 0)

CELLPOSE_4X = CELLPOSE_VERSION >= (4, 0)


def initialize_cellpose_model(model_type: str, gpu: bool = False) -> CellposeModel:
    """Initialize a CellposeModel with version-aware configuration.

    Handles differences between Cellpose 3.x and 4.x APIs and validates
    model compatibility with the installed Cellpose version.

    Args:
        model_type (str): Cellpose model type to use (e.g., 'cyto3', 'nuclei', 'cpsam')
            or a path to a custom trained model (e.g., 'models/my_custom_model').
            - Cellpose 3.x model_type: Supports 'cyto3', 'nuclei', 'cyto2'
            - Cellpose 4.x: Only supports 'cpsam'
            - Custom model paths (containing path separators) are supported in both versions
        gpu (bool, optional): Whether to use GPU for inference. Default is False.

    Returns:
        CellposeModel: Initialized Cellpose model ready for inference.

    Raises:
        ValueError: If model_type is incompatible with installed Cellpose version.
    """
    # Check if model_type is a custom model path (contains path separators)
    is_custom_model = model_type is not None and (
        "/" in model_type or "\\" in model_type
    )

    # Validate compatibility
    # Custom model paths are allowed with any Cellpose version
    if CELLPOSE_4X and model_type not in ("cpsam", None) and not is_custom_model:
        raise ValueError(
            f"Model '{model_type}' requires Cellpose 3.x. "
            f"Cellpose 4.x only supports the 'cpsam' model. "
            f"Either change your config to use model='cpsam', "
            f"or downgrade Cellpose: uv pip install cellpose==3.1.0"
        )
    if not CELLPOSE_4X and model_type == "cpsam":
        raise ValueError(
            f"CPSAM model requires Cellpose 4.x. "
            f"You have Cellpose {'.'.join(map(str, CELLPOSE_VERSION))}. "
            f"Upgrade with: uv pip install cellpose==4.0.4 torch==2.7.0 torchvision==0.22.0"
        )

    # Version-aware initialization
    # Custom model paths use pretrained_model parameter in both versions
    if CELLPOSE_4X:
        return CellposeModel(pretrained_model=model_type, gpu=gpu)
    elif is_custom_model:
        # For Cellpose 3.x with custom models, use pretrained_model
        return CellposeModel(pretrained_model=model_type, gpu=gpu)
    else:
        return CellposeModel(model_type=model_type, gpu=gpu)


def segment_cellpose(
    data,
    dapi_index,
    cyto_index,
    primary_diameter,
    cell_diameter,
    cellpose_model="cyto3",
    helper_index=None,
    cellpose_kwargs=dict(
        flow_threshold=0.4,
        cellprob_threshold=0,
        primary_flow_threshold=None,
        primary_cellprob_threshold=None,
        cell_flow_threshold=None,
        cell_cellprob_threshold=None,
    ),
    cells=True,
    reconcile="consensus",
    logscale=True,
    return_counts=False,
    gpu=False,
):
    """Segment cells using Cellpose algorithm.

    Args:
        data (numpy.ndarray): Multichannel image data.
        dapi_index (int): Index of DAPI channel.
        cyto_index (int): Index of cytoplasmic channel.
        primary_diameter (int): Estimated diameter of primary.
        cell_diameter (int): Estimated diameter of cells.
        cellpose_model (str, optional): Cellpose model type to use (e.g., 'cyto3', 'cpsam').
            Default is 'cyto3'. Use 'cpsam' for Cellpose-SAM (requires Cellpose 4.x).
        helper_index (int, optional): Index of helper channel for improved segmentation (CPSAM feature).
            Only used with multi-channel models. Default is None (blank channel).
        cellpose_kwargs (dict, optional): Additional keyword arguments for Cellpose, including:
            - flow_threshold (float): Default flow threshold for both primary and cells if specific ones not provided
            - cellprob_threshold (float): Default cell probability threshold for both primary and cells if specific ones not provided
            - primary_flow_threshold (float): Specific flow threshold for primary segmentation
            - primary_cellprob_threshold (float): Specific cell probability threshold for primary segmentation
            - cell_flow_threshold (float): Specific flow threshold for cell segmentation
            - cell_cellprob_threshold (float): Specific cell probability threshold for cell segmentation
        cells (bool, optional): Whether to segment both primary and cells or just primary.
        reconcile (str, optional): Method for reconciling primary and cells. Default is 'consensus'.
        logscale (bool, optional): Whether to apply logarithmic transformation to image data.
        return_counts (bool, optional): Whether to return counts of primary and cells. Default is False.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.

    Returns:
        tuple or numpy.ndarray: If 'cells' is True, returns tuple of primary and cell segmentation masks,
        otherwise returns only primary segmentation mask. If return_counts is True, includes a dictionary of counts.
    """
    # Extract log_kwargs from cellpose_kwargs
    log_kwargs = cellpose_kwargs.pop("log_kwargs", dict())

    # Extract specific thresholds for primary and cells
    primary_flow_threshold = cellpose_kwargs.pop(
        "primary_flow_threshold", cellpose_kwargs.get("flow_threshold", 0.4)
    )
    primary_cellprob_threshold = cellpose_kwargs.pop(
        "primary_cellprob_threshold", cellpose_kwargs.get("cellprob_threshold", 0)
    )
    cell_flow_threshold = cellpose_kwargs.pop(
        "cell_flow_threshold", cellpose_kwargs.get("flow_threshold", 0.4)
    )
    cell_cellprob_threshold = cellpose_kwargs.pop(
        "cell_cellprob_threshold", cellpose_kwargs.get("cellprob_threshold", 0)
    )

    # Create separate kwargs dictionaries
    primary_kwargs = {
        "flow_threshold": primary_flow_threshold,
        "cellprob_threshold": primary_cellprob_threshold,
    }

    cell_kwargs = {
        "flow_threshold": cell_flow_threshold,
        "cellprob_threshold": cell_cellprob_threshold,
    }

    # Prepare data for Cellpose by creating a merged RGB image
    rgb = prepare_cellpose(
        data,
        dapi_index,
        cyto_index,
        helper_index=helper_index,
        logscale=logscale,
        log_kwargs=log_kwargs,
    )

    counts = {}

    # Perform cell segmentation using Cellpose
    if cells:
        if return_counts:
            primary, cells, seg_counts = segment_cellpose_rgb(
                rgb,
                primary_diameter,
                cell_diameter,
                cellpose_model=cellpose_model,
                reconcile=reconcile,
                return_counts=True,
                gpu=gpu,
                primary_kwargs=primary_kwargs,
                cell_kwargs=cell_kwargs,
            )
            counts.update(seg_counts)

        else:
            primary, cells = segment_cellpose_rgb(
                rgb,
                primary_diameter,
                cell_diameter,
                cellpose_model=cellpose_model,
                reconcile=reconcile,
                gpu=gpu,
                primary_kwargs=primary_kwargs,
                cell_kwargs=cell_kwargs,
            )

        counts["final_primary"] = len(np.unique(primary)) - 1
        counts["final_cells"] = len(np.unique(cells)) - 1
        counts_df = pd.DataFrame([counts])
        print(f"Number of primary segmented: {counts['final_primary']}")
        print(f"Number of cells segmented: {counts['final_cells']}")

        if return_counts:
            return primary, cells, counts_df
        else:
            return primary, cells
    else:
        primary = segment_cellpose_primary_rgb(
            rgb,
            primary_diameter,
            cellpose_model=cellpose_model,
            gpu=gpu,
            **primary_kwargs,
        )
        counts["final_primary"] = len(np.unique(primary)) - 1
        print(f"Number of primary segmented: {counts['final_primary']}")
        counts_df = pd.DataFrame([counts])

        if return_counts:
            return primary, counts_df
        else:
            return primary


def prepare_cellpose(
    data, dapi_index, cyto_index, helper_index=None, logscale=True, log_kwargs=dict()
):
    """Prepare a three-channel RGB image for use with Cellpose.

    Args:
        data (list or numpy.ndarray): List or array containing DAPI and cytoplasmic channel images.
        dapi_index (int): Index of the DAPI channel in the data.
        cyto_index (int): Index of the cytoplasmic channel in the data.
        helper_index (int, optional): Index of helper channel for improved segmentation.
            If None, uses blank (zeros) channel. Default is None.
        logscale (bool, optional): Whether to apply log scaling to the cytoplasmic and helper channels.
            Default is True.
        log_kwargs (dict, optional): Additional keyword arguments for log scaling.

    Returns:
        numpy.ndarray: Three-channel RGB image. Red is the helper channel (or blank if helper_index=None),
            green is the cytoplasmic channel, and blue is the DAPI channel.
    """
    # Extract DAPI and cytoplasmic channel images from the data
    dapi = data[dapi_index]
    cyto = data[cyto_index]

    # Extract helper channel if provided, otherwise use blank channel
    if helper_index is not None:
        helper = data[helper_index]
    else:
        helper = np.zeros_like(cyto)

    # Apply log scaling to the cytoplasmic and helper channels if specified
    if logscale:
        cyto = image_log_scale(cyto, **log_kwargs)
        # Safe normalization: check for zero max before division
        if cyto.max() > 0:
            cyto = cyto / cyto.max()

        # Apply log scaling to helper channel too
        helper = image_log_scale(helper, **log_kwargs)
        if helper.max() > 0:
            helper = helper / helper.max()

    # Normalize the intensity of the DAPI channel and scale it to the range [0, 1]
    dapi_upper = np.percentile(dapi, 99.5)
    # Safe normalization: check for zero before division
    if dapi_upper > 0:
        dapi = dapi / dapi_upper
    dapi = np.clip(dapi, 0, 1)

    # Convert the channels to uint8 format for RGB image creation
    red, green, blue = img_as_ubyte(helper), img_as_ubyte(cyto), img_as_ubyte(dapi)

    # Stack the channels to create the RGB image: [helper/red, cyto/green, dapi/blue]
    return np.array([red, green, blue])


def estimate_diameters(
    data,
    dapi_index,
    cyto_index,
    helper_index=None,
    channels=[2, 3],  # Default channels for cell estimation
    cellpose_model="cyto3",
    cellpose_kwargs=dict(flow_threshold=0.4, cellprob_threshold=0),
    gpu=False,
    logscale=True,
):
    """Estimate optimal cell diameter using Cellpose's diameter estimation.

    Note: Only supported with Cellpose 3.x. Cellpose 4.x does not have SizeModel.
          With Cellpose 4.x, you must specify primary_diameter and cell_diameter in config.

    Args:
        data (numpy.ndarray): Multichannel image data
        dapi_index (int): Index of DAPI channel
        cyto_index (int): Index of cytoplasmic channel
        helper_index (int, optional): Index of helper channel. Default is None.
        channels (list): Channel indices for diameter estimation [cytoplasm, primary]
        cellpose_model (str): Cellpose model type to use (e.g., 'cyto3', 'cpsam')
        cellpose_kwargs (dict): Additional keyword arguments for Cellpose
        gpu (bool): Whether to use GPU
        logscale (bool): Whether to apply log scaling to image

    Returns:
        tuple: Estimated diameters for (primary, cells)

    Raises:
        NotImplementedError: If using Cellpose 4.x (SizeModel not available)
    """
    # Check version compatibility
    if CELLPOSE_4X:
        raise NotImplementedError(
            "Automatic diameter estimation is not supported with Cellpose 4.x. "
            "SizeModel is not available in Cellpose 4.x. "
            "Please specify primary_diameter and cell_diameter explicitly in your config, "
            "or downgrade to Cellpose 3.x: uv pip install cellpose==3.1.0"
        )

    # Import SizeModel only for Cellpose 3.x (doesn't exist in 4.x)
    from cellpose.models import SizeModel

    # Prepare RGB image
    log_kwargs = cellpose_kwargs.pop(
        "log_kwargs", dict()
    )  # Extract log_kwargs from cellpose_kwargs
    rgb = prepare_cellpose(
        data,
        dapi_index,
        cyto_index,
        helper_index=helper_index,
        logscale=logscale,
        log_kwargs=log_kwargs,
    )

    # Find optimal primary diameter using explicit SizeModel
    print("Estimating primary diameters...")
    model_primary = CellposeModel(model_type="nuclei", gpu=gpu)
    size_model_primary = SizeModel(
        cp_model=model_primary, pretrained_size=cellpose_models.size_model_path("nuclei")
    )
    diam_nuclear, _ = size_model_primary.eval(rgb, channels=[3, 0])
    diam_nuclear = np.maximum(5.0, diam_nuclear)
    diam_nuclear = float(diam_nuclear)
    print(f"Estimated nuclear diameter: {diam_nuclear:.1f} pixels")

    # Find optimal cell diameter using explicit SizeModel
    print("Estimating cell diameters...")
    model_cyto = CellposeModel(model_type=cellpose_model, gpu=gpu)
    size_model_cyto = SizeModel(
        cp_model=model_cyto,
        pretrained_size=cellpose_models.size_model_path(cellpose_model),
    )
    diam_cell, _ = size_model_cyto.eval(rgb, channels=channels)
    diam_cell = np.maximum(5.0, diam_cell)
    diam_cell = float(diam_cell)
    print(f"Estimated cell diameter: {diam_cell:.1f} pixels")

    return diam_nuclear, diam_cell


def segment_cellpose_rgb(
    rgb,
    primary_diameter,
    cell_diameter,
    cellpose_model="cyto3",
    reconcile="consensus",
    remove_edges=True,
    return_counts=False,
    gpu=False,
    primary_kwargs=None,
    cell_kwargs=None,
    **kwargs,
):
    """Segment primary and cells using the Cellpose algorithm from an RGB image.

    Args:
        rgb (numpy.ndarray): RGB image.
        primary_diameter (int): Diameter of primary for segmentation.
        cell_diameter (int): Diameter of cells for segmentation.
        cellpose_model (str, optional): Cellpose model type to use (e.g., 'cyto3', 'cpsam').
            Default is 'cyto3'. Cellpose 3.x model_type: Use 'cyto3', 'nuclei', 'cyto2'.
            Cellpose 4.x: Use 'cpsam' only.
        reconcile (str, optional): Method for reconciling primary and cells. Default is 'consensus'.
        remove_edges (bool, optional): Whether to remove primary and cells touching the image edges. Default is True.
        return_counts (bool, optional): Whether to return counts of primary and cells before reconciliation. Default is False.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        primary_kwargs (dict, optional): Specific parameters for primary segmentation. Default is None.
        cell_kwargs (dict, optional): Specific parameters for cell segmentation. Default is None.
        kwargs: Additional keyword arguments applied to both primary and cell segmentation if specific kwargs not provided.

    Returns:
        tuple: A tuple containing:
            - primary (numpy.ndarray): Labeled segmentation mask of primary.
            - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
            - (optional) counts (dict): Counts of primary and cells at different stages if return_counts is True.

    Raises:
        ValueError: If model is incompatible with installed Cellpose version
    """
    # Create Cellpose models using version-aware helper
    # primary_model_type: "cpsam" for 4.x, "nuclei" for 3.x
    primary_model_type = "cpsam" if CELLPOSE_4X else "nuclei"
    model_dapi = initialize_cellpose_model(primary_model_type, gpu=gpu)
    model_cyto = initialize_cellpose_model(cellpose_model, gpu=gpu)

    # Set default kwargs if not provided
    if primary_kwargs is None:
        primary_kwargs = kwargs.copy()
    if cell_kwargs is None:
        cell_kwargs = kwargs.copy()

    counts = {}

    # Segment primary using primary-specific parameters
    # Pass only blue channel (DAPI) for primary segmentation
    primary, _, _ = model_dapi.eval(rgb[2], diameter=primary_diameter, **primary_kwargs)

    # Segment cells using cell-specific parameters
    # For CPSAM (Cellpose 4.x), use all 3 channels: [red/helper, green/cyto, blue/DAPI]
    # For standard models (Cellpose 3.x), use [green/cyto, blue/DAPI]
    if CELLPOSE_4X and cellpose_model == "cpsam":
        # CPSAM can use all 3 channels
        cells, _, _ = model_cyto.eval(
            rgb, diameter=cell_diameter, channels=[1, 2, 3], **cell_kwargs
        )
    else:
        # Standard cyto models use cytoplasm (green=index 2) and primary (blue=index 3) channels
        cells, _, _ = model_cyto.eval(
            rgb, diameter=cell_diameter, channels=[2, 3], **cell_kwargs
        )

    counts["initial_primary"] = (
        len(np.unique(primary)) - 1
    )  # Subtract 1 to exclude background
    counts["initial_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['initial_primary']} primary before removing edges",
        file=sys.stderr,
    )
    print(
        f"found {counts['initial_cells']} cells before removing edges", file=sys.stderr
    )

    # Remove primary and cells touching the image edges if specified
    if remove_edges:
        print("removing edges")
        primary = clear_border(primary)
        cells = clear_border(cells)

    counts["after_edge_removal_primary"] = len(np.unique(primary)) - 1
    counts["after_edge_removal_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['after_edge_removal_primary']} primary before reconciling",
        file=sys.stderr,
    )
    print(
        f"found {counts['after_edge_removal_cells']} cells before reconciling",
        file=sys.stderr,
    )

    # Reconcile primary and cells if specified
    if reconcile:
        print(f"reconciling masks with method how={reconcile}")
        primary, cells = reconcile_primary_cells(primary, cells, how=reconcile)

    counts["final_cells"] = len(np.unique(cells)) - 1
    print(
        f"found {counts['final_cells']} primary/cells after reconciling", file=sys.stderr
    )

    if return_counts:
        return primary, cells, counts
    else:
        return primary, cells


def segment_cellpose_primary_rgb(
    rgb,
    primary_diameter,
    cellpose_model="nuclei",
    gpu=False,
    remove_edges=True,
    **kwargs,
):
    """Segment primary using the Cellpose algorithm from an RGB image.

    Args:
        rgb (numpy.ndarray): RGB image.
        primary_diameter (int): Diameter of primary for segmentation.
        cellpose_model (str, optional): Cellpose model_type to use. Default is "nuclei".
            Can also use "cyto3" or other models if the "nuclei" model_type produces poor results.
        gpu (bool, optional): Whether to use GPU for segmentation. Default is False.
        remove_edges (bool, optional): Whether to remove primary touching the image edges. Default is True.
        **kwargs: Additional keyword arguments.

    Returns:
        numpy.ndarray: Labeled segmentation mask of primary.
    """
    # Create Cellpose model using version-aware helper
    model = initialize_cellpose_model(cellpose_model, gpu=gpu)

    # Segment primary using CellposeModel from the RGB image
    # Pass only blue channel (DAPI) for primary segmentation
    primary, _, _ = model.eval(rgb[2], diameter=primary_diameter, **kwargs)

    # Print the number of primary found before and after removing edges
    print(
        f"found {len(np.unique(primary))} primary before removing edges", file=sys.stderr
    )

    # Remove primary touching the image edges if specified
    if remove_edges:
        print("removing edges")
        primary = clear_border(primary)

    # Print the final number of primary after processing
    print(f"found {len(np.unique(primary))} final primary", file=sys.stderr)

    # Return the segmented primary
    return primary
