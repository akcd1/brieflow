"""Function for identifying and isolating the cytoplasm region."""

import numpy as np


def identify_cytoplasm_cellpose(primary, cells):
    """Identifies and isolates the cytoplasm region in an image based on the provided primary object and cells masks.

    Args:
        primary (ndarray): A 2D array representing the primary object regions.
        cells (ndarray): A 2D array representing the cells regions.

    Returns:
        ndarray: A 2D array representing the cytoplasm regions.
    """
    # Check if the number of unique labels in primary and cells are the same
    if len(np.unique(primary)) != len(np.unique(cells)):
        return None  # Break out of the function if the masks are not compatible

    # Create an empty cytoplasmic mask with the same shape as cells
    cytoplasms = np.zeros(cells.shape)

    # Iterate over each unique cell label
    for cell_label in np.unique(cells):
        # Skip if the cell label is 0 (background)
        if cell_label == 0:
            continue

        # Find the corresponding primary object label for this cell
        primary_label = cell_label

        # Get the coordinates of the primary object and cell regions
        primary_coords = np.argwhere(primary == primary_label)
        cell_coords = np.argwhere(cells == cell_label)

        # Update the cytoplasmic mask with the cell region
        cytoplasms[cell_coords[:, 0], cell_coords[:, 1]] = cell_label

        # Remove the primary object region from the cytoplasmic mask
        cytoplasms[primary_coords[:, 0], primary_coords[:, 1]] = 0

    # Calculate the number of identified cytoplasms (excluding background label)
    num_cytoplasm_segmented = len(np.unique(cytoplasms)) - 1
    print(f"Number of cytoplasms identified: {num_cytoplasm_segmented}")

    # Return the final cytoplasm array
    return cytoplasms.astype(int)
