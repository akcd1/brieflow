"""StarDist-based Image Segmentation!

This module provides functions for segmenting microscopy images using the StarDist algorithm
(relating to SBS base calling and phenotyping -- steps 1 and 2). It includes functions for:

1. Cell and Primary Segmentation: Segmenting both cellular components using StarDist
2. Image Preprocessing: Applying intensity normalization and preprocessing techniques
3. Label Reconciliation: Reconciling primary object and cell labels based on their spatial relationships
4. Mask Processing: Manipulating and refining segmentation masks
5. Utility Functions: Supporting operations for image analysis and segmentation tasks
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, Union

from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.segmentation import clear_border

from lib.shared.segmentation_utils import reconcile_primary_cells


def segment_stardist(
    data,
    dapi_index,
    cyto_index,
    model_type="2D_versatile_fluo",
    stardist_kwargs=dict(
        prob_threshold=0.479071,
        nms_threshold=0.3,
        primary_prob_threshold=None,
        primary_nms_threshold=None,
        cell_prob_threshold=None,
        cell_nms_threshold=None,
    ),
    cells=True,
    reconcile="consensus",
    return_counts=False,
    gpu=False,
):
    """Segment cells using StarDist algorithm with separate parameters for primary objects and cells.

    Args:
        data: Multichannel image data
        dapi_index: Index of DAPI channel
        cyto_index: Index of cytoplasmic channel
        model_type: StarDist model type to use
        stardist_kwargs: Additional keyword arguments for StarDist, including:
            - prob_threshold: Default probability threshold for both primary objects and cells
            - nms_threshold: Default NMS threshold for both primary objects and cells
            - primary_prob_threshold: Specific probability threshold for primary segmentation
            - primary_nms_threshold: Specific NMS threshold for primary segmentation
            - cell_prob_threshold: Specific probability threshold for cell segmentation
            - cell_nms_threshold: Specific NMS threshold for cell segmentation
        cells: Whether to segment both primary objects and cells or just primary objects
        reconcile: Method for reconciling primary objects and cells
        return_counts: Whether to return counts of primary objects and cells
        gpu: Whether to use GPU for segmentation

    Returns:
        Segmentation masks with optional counts
    """
    # Extract specific thresholds for primary objects and cells
    primary_prob_threshold = stardist_kwargs.pop(
        "primary_prob_threshold", stardist_kwargs.get("prob_threshold", 0.479071)
    )
    primary_nms_threshold = stardist_kwargs.pop(
        "primary_nms_threshold", stardist_kwargs.get("nms_threshold", 0.3)
    )
    cell_prob_threshold = stardist_kwargs.pop(
        "cell_prob_threshold", stardist_kwargs.get("prob_threshold", 0.479071)
    )
    cell_nms_threshold = stardist_kwargs.pop(
        "cell_nms_threshold", stardist_kwargs.get("nms_threshold", 0.3)
    )

    # Create separate kwargs dictionaries
    primary_kwargs = {
        "prob_thresh": primary_prob_threshold,
        "nms_thresh": primary_nms_threshold,
    }
    cell_kwargs = {
        "prob_thresh": cell_prob_threshold,
        "nms_thresh": cell_nms_threshold,
    }

    # Prepare channels for StarDist
    dapi = prepare_channel(data[dapi_index])
    cyto = prepare_channel(data[cyto_index])

    counts = {}

    # Perform cell segmentation using StarDist
    if cells:
        if return_counts:
            primary, cells, seg_counts = segment_stardist_multichannel(
                dapi,
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                return_counts=True,
                gpu=gpu,
                primary_kwargs=primary_kwargs,
                cell_kwargs=cell_kwargs,
            )
            counts.update(seg_counts)

        else:
            primary, cells = segment_stardist_multichannel(
                dapi,
                cyto,
                model_type=model_type,
                reconcile=reconcile,
                gpu=gpu,
                primary_kwargs=primary_kwargs,
                cell_kwargs=cell_kwargs,
            )

        counts["final_primary"] = len(np.unique(primary)) - 1
        counts["final_cells"] = len(np.unique(cells)) - 1
        counts_df = pd.DataFrame([counts])
        print(f"Number of primary objects segmented: {counts['final_primary']}")
        print(f"Number of cells segmented: {counts['final_cells']}")

        if return_counts:
            return primary, cells, counts_df
        else:
            return primary, cells
    else:
        primary = segment_stardist_primary(
            dapi, model_type=model_type, gpu=gpu, **primary_kwargs
        )

        counts["final_primary"] = len(np.unique(primary)) - 1
        print(f"Number of primary objects segmented: {counts['final_primary']}")

        counts_df = pd.DataFrame([counts])

        if return_counts:
            return primary, counts_df
        else:
            return primary


def prepare_channel(data):
    """Prepare channel data for segmentation using StarDist's normalization.

    Args:
        data: Input channel data

    Returns:
        Processed channel data
    """
    # Use StarDist's recommended normalization
    return normalize(data, 1, 99.8, axis=None)


def segment_stardist_multichannel(
    dapi,
    cyto,
    model_type="2D_versatile_fluo",
    reconcile="consensus",
    remove_edges=True,
    return_counts=False,
    gpu=False,
    primary_kwargs=None,
    cell_kwargs=None,
    **kwargs,
):
    """Segment primary objects and cells using the StarDist algorithm with separate parameters.

    Args:
        dapi: DAPI channel data
        cyto: Cytoplasmic channel data
        model_type: StarDist model type to use
        reconcile: Method for reconciling primary objects and cells
        remove_edges: Whether to remove edges from the masks
        return_counts: Whether to return counts of primary objects and cells
        gpu: Whether to use GPU for segmentation
        primary_kwargs: Specific parameters for primary segmentation
        cell_kwargs: Specific parameters for cell segmentation
        kwargs: Additional keyword arguments applied to both if specific kwargs not provided

    Returns:
        tuple: A tuple containing:
            - primary (numpy.ndarray): Labeled segmentation mask of primary objects.
            - cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
            - (optional) counts (dict): Counts of primary objects and cells at different stages if return_counts is True.
    """
    # Initialize StarDist models for primary and cytoplasmic segmentation
    model_primary = StarDist2D.from_pretrained(model_type)
    model_cells = StarDist2D.from_pretrained(model_type)

    # Set default kwargs if not provided
    if primary_kwargs is None:
        primary_kwargs = kwargs.copy()
    if cell_kwargs is None:
        cell_kwargs = kwargs.copy()

    counts = {}

    if gpu:
        model_primary.config.use_gpu = True
        model_cells.config.use_gpu = True

    # Segment primary objects using primary-specific parameters
    primary, _ = model_primary.predict_instances(dapi, **primary_kwargs)

    # Segment cells using cell-specific parameters
    cells, _ = model_cells.predict_instances(cyto, **cell_kwargs)

    counts["initial_primary"] = len(np.unique(primary)) - 1
    counts["initial_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['initial_primary']} primary objects before removing edges",
        file=sys.stderr,
    )
    print(
        f"found {counts['initial_cells']} cells before removing edges", file=sys.stderr
    )

    if remove_edges:
        print("removing edges")
        primary = clear_border(primary)
        cells = clear_border(cells)

    counts["after_edge_removal_primary"] = len(np.unique(primary)) - 1
    counts["after_edge_removal_cells"] = len(np.unique(cells)) - 1

    print(
        f"found {counts['after_edge_removal_primary']} primary objects before reconciling",
        file=sys.stderr,
    )
    print(
        f"found {counts['after_edge_removal_cells']} cells before reconciling",
        file=sys.stderr,
    )

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


def segment_stardist_primary(
    dapi,
    model_type="2D_versatile_fluo",
    gpu=False,
    remove_edges=True,
    **kwargs,
):
    """Segment primary objects using the StarDist algorithm.

    Args:
        dapi: DAPI channel data
        model_type: StarDist model type to use
        remove_edges: Whether to remove edges from the masks
        gpu: Whether to use GPU for segmentation
        **kwargs: Parameters for StarDist segmentation including:
                 - prob_thresh: Probability threshold for segmentation
                 - nms_thresh: Non-maximum suppression threshold for segmentation
    Returns:
        Segmented primary object masks
    """
    # Initialize StarDist model
    model = StarDist2D.from_pretrained(model_type)
    if gpu:
        model.config.use_gpu = True

    # Segment primary objects with specified parameters
    primary, _ = model.predict_instances(dapi, **kwargs)

    print(
        f"found {len(np.unique(primary))} primary objects before removing edges", file=sys.stderr
    )

    if remove_edges:
        print("removing edges")
        primary = clear_border(primary)

    print(f"found {len(np.unique(primary))} final primary objects", file=sys.stderr)

    return primary
