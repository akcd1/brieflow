"""Debug visualization module for phenotype channel alignment.

Provides functions to visualize alignment quality across multiple tiles
by loading aligned images directly from the output folder.
"""

import numpy as np
from pathlib import Path
import tifffile


def visualize_alignment_debug(
    images_folder,
    channel_names,
    viz_channels,
    crop_size=500,
    n_tiles=16,
    tile_range=None,
    well_filter=None,
    seed=42,
):
    """Visualize alignment quality across multiple tiles with one center crop per tile.

    Loads aligned images from a folder and displays center crops in a grid.
    First channel shown in grayscale with remaining 3 channels as RGB overlay.
    Color fringing indicates misalignment.

    Args:
        images_folder (str): Path to folder containing *__aligned.tiff files.
        channel_names (list): List of all channel names in the images.
        viz_channels (list): List of 4 channel names to visualize
            (1st=grayscale, 2nd-4th=RGB overlay).
        crop_size (int, optional): Size of center crop in pixels. Defaults to 500.
        n_tiles (int, optional): Number of tiles to display. Defaults to 16.
        tile_range (list of int, optional): Specific tile indices to display in order.
            If provided, overrides random selection. Must not exceed n_tiles.
            Defaults to None (random selection).
        well_filter (str, optional): Filter for specific well (e.g., "A1"). Defaults to None.
        seed (int, optional): Random seed for reproducible tile selection. Defaults to 42.

    Returns:
        matplotlib.figure.Figure: Figure with grid of alignment visualizations,
            or None if there's an error.

    Example:
        >>> fig = visualize_alignment_debug(
        ...     images_folder="/path/to/brieflow_output/phenotype/images",
        ...     channel_names=["Tubulin", "NHS_ester", "CDPK1", "DAPI"],
        ...     viz_channels=["Tubulin", "NHS_ester", "CDPK1", "DAPI"],
        ...     crop_size=500,
        ...     n_tiles=16,
        ...     well_filter="A1"
        ... )
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import re

    if len(viz_channels) != 4:
        print(
            f"Error: Need exactly 4 channels (1 grayscale + 3 RGB), got {len(viz_channels)}"
        )
        return None

    # Find all aligned images
    folder = Path(images_folder)
    aligned_files = sorted(
        folder.glob("*__aligned.tiff"),
        key=lambda f: int(re.search(r"_T-(\d+)_", f.name).group(1))
        if re.search(r"_T-(\d+)_", f.name)
        else 0,
    )

    if not aligned_files:
        print(f"Error: No *__aligned.tiff files found in {images_folder}")
        return None

    # Apply well filter if specified
    if well_filter is not None:
        aligned_files = [f for f in aligned_files if f"_W-{well_filter}_" in f.name]
        if not aligned_files:
            print(f"Error: No files found for well {well_filter}")
            return None

    # Select tiles based on tile_range or randomly
    n_available = len(aligned_files)

    if tile_range is not None:
        # Use specified range
        if len(tile_range) > n_tiles:
            print(
                f"Error: tile_range specifies {len(tile_range)} tiles but n_tiles={n_tiles}"
            )
            return None

        # Validate indices
        invalid_indices = [i for i in tile_range if i < 0 or i >= n_available]
        if invalid_indices:
            print(
                f"Error: tile_range contains invalid indices {invalid_indices}. Valid range: 0-{n_available - 1}"
            )
            return None

        selected_indices = tile_range
        selected_files = [aligned_files[i] for i in selected_indices]
        n_to_show = len(tile_range)
    else:
        # Randomly select tiles
        np.random.seed(seed)
        n_to_show = min(n_tiles, n_available)
        selected_indices = np.random.choice(n_available, n_to_show, replace=False)
        selected_files = [aligned_files[i] for i in sorted(selected_indices)]

    print(f"Visualizing {n_to_show} tiles from {n_available} available...")

    # Get channel indices
    channel_indices = []
    for ch_name in viz_channels:
        if ch_name not in channel_names:
            print(f"Error: Channel '{ch_name}' not found in {channel_names}")
            return None
        channel_indices.append(channel_names.index(ch_name))

    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_to_show)))
    n_rows = int(np.ceil(n_to_show / n_cols))

    # Create figure
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)

    for idx, filepath in enumerate(selected_files):
        # Load image
        aligned_data = tifffile.imread(filepath)

        # Handle different array shapes
        if aligned_data.ndim == 2:
            print(f"Warning: {filepath.name} is 2D, skipping")
            continue
        elif aligned_data.ndim == 4:
            # (stack, channel, y, x) -> take max projection
            aligned_data = aligned_data.max(axis=0)

        _, height, width = aligned_data.shape

        # Calculate center crop coordinates
        y_start = (height - crop_size) // 2
        x_start = (width - crop_size) // 2
        y_end = y_start + crop_size
        x_end = x_start + crop_size

        # Ensure crop is within bounds
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        y_end = min(height, y_end)
        x_end = min(width, x_end)
        actual_crop_h = y_end - y_start
        actual_crop_w = x_end - x_start

        # Create combined RGBA image
        rgba = np.zeros((actual_crop_h, actual_crop_w, 3))

        # Add grayscale (first channel) as base layer
        gray_crop = aligned_data[channel_indices[0], y_start:y_end, x_start:x_end]
        p2, p98 = np.percentile(gray_crop, [2, 98])
        gray_norm = np.clip((gray_crop - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Add RGB composite (channels 2-4) overlaid on grayscale
        rgb = np.zeros((actual_crop_h, actual_crop_w, 3))
        for i, ch_idx in enumerate(channel_indices[1:]):
            crop = aligned_data[ch_idx, y_start:y_end, x_start:x_end]
            p2, p98 = np.percentile(crop, [2, 98])
            crop_norm = np.clip((crop - p2) / (p98 - p2 + 1e-8), 0, 1)
            rgb[:, :, i] = crop_norm

        # Blend: 50% grayscale, 50% RGB
        for i in range(3):
            rgba[:, :, i] = 0.5 * gray_norm + 0.5 * rgb[:, :, i]

        # Extract tile ID from filename
        tile_id = filepath.stem.replace("__aligned", "")

        ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols])
        ax.imshow(rgba)
        ax.set_title(
            f"{tile_id}\n"
            + f"Gray: {viz_channels[0]} | R={viz_channels[1]}, G={viz_channels[2]}, B={viz_channels[3]}",
            fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout()
    return fig
