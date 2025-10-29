"""Improved vacuole segmentation with enhanced declumping strategies.

Key improvements for handling clumping:
1. Multi-scale approach with adaptive thresholding
2. Enhanced watershed with multiple marker strategies
3. Morphological operations for better separation
4. H-minima transform to suppress spurious local maxima
5. Alternative declumping with distance + intensity features
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation, feature, exposure
import cv2


def get_feret_diameters(coords):
    """Compute the minimum and maximum Feret diameters of a 2D shape."""
    cnt = coords.astype(np.int32)
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    return min(w, h), max(w, h)


def enhanced_declumping(binary_mask, vacuole_img, min_distance=20, h_minima=None):
    """
    Enhanced declumping using multiple strategies.
    
    Parameters
    ----------
    binary_mask : ndarray
        Binary mask of vacuoles
    vacuole_img : ndarray
        Original intensity image
    min_distance : int
        Minimum distance between peaks
    h_minima : float, optional
        Height threshold for h-minima transform (suppresses small local maxima)
        If None, automatically calculated as a fraction of distance transform range
    
    Returns
    -------
    declumped : ndarray
        Labeled segmentation mask
    """
    # Strategy 1: Distance transform with adaptive h-minima
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Auto-calculate h_minima if not provided (suppress shallow local maxima)
    if h_minima is None:
        h_minima = np.percentile(distance[distance > 0], 20)  # Suppress bottom 20% of peaks
    
    # Apply h-minima transform to suppress spurious local maxima
    # This reduces over-segmentation by merging shallow peaks
    distance_filtered = morphology.h_minima(distance, h_minima)
    
    # Find local maxima with increased minimum distance
    local_max = feature.peak_local_max(
        distance_filtered,
        min_distance=min_distance,
        labels=binary_mask,
        exclude_border=True  # Exclude border to avoid edge artifacts
    )
    
    # Strategy 2: If we have intensity image, use intensity peaks as additional markers
    # This helps separate vacuoles that are close but have distinct intensity peaks
    if vacuole_img is not None:
        # Smooth intensity image to find robust peaks
        smoothed_intensity = filters.gaussian(vacuole_img, sigma=2)
        
        # Find intensity peaks within the mask
        intensity_peaks = feature.peak_local_max(
            smoothed_intensity,
            min_distance=min_distance // 2,  # Allow closer spacing for intensity peaks
            labels=binary_mask,
            exclude_border=True,
            threshold_rel=0.3  # Only consider peaks above 30% of max
        )
        
        # Combine distance and intensity markers
        # Use a set to avoid duplicates (peaks that are close to each other)
        all_peaks = []
        for peak in local_max:
            all_peaks.append(tuple(peak))
        
        for peak in intensity_peaks:
            # Only add intensity peak if it's not too close to existing distance peak
            is_far_enough = True
            for existing_peak in all_peaks:
                dist = np.sqrt((peak[0] - existing_peak[0])**2 + (peak[1] - existing_peak[1])**2)
                if dist < min_distance / 2:
                    is_far_enough = False
                    break
            if is_far_enough:
                all_peaks.append(tuple(peak))
        
        markers_array = np.array(all_peaks)
    else:
        markers_array = local_max
    
    # Create markers for watershed
    markers = np.zeros_like(binary_mask, dtype=int)
    if len(markers_array) > 0:
        markers[tuple(markers_array.T)] = np.arange(1, len(markers_array) + 1)
        
        # Apply watershed on negative distance (valleys become peaks)
        declumped = segmentation.watershed(-distance, markers, mask=binary_mask)
        
        # Recover unassigned regions
        missing = (declumped == 0) & binary_mask
        if np.any(missing):
            labeled_missing, _ = ndimage.label(missing)
            labeled_missing[labeled_missing > 0] += declumped.max()
            declumped += labeled_missing
    else:
        # Fallback to simple connected components
        declumped, _ = ndimage.label(binary_mask)
    
    return declumped


def apply_morphological_opening(binary_mask, opening_disk_radius=1):
    """
    Apply morphological opening to separate weakly connected vacuoles.
    
    Parameters
    ----------
    binary_mask : ndarray
        Binary mask of vacuoles
    opening_disk_radius : int
        Radius of disk structuring element in pixels.
        - radius=1: Cuts bridges 1-2 pixels wide (default, very thin)
        - radius=2: Cuts bridges 2-4 pixels wide (thin)
        - radius=3: Cuts bridges 3-6 pixels wide (medium)
        - radius=4: Cuts bridges 4-8 pixels wide (thick)
        Larger radius = more aggressive bridge cutting
    
    Returns
    -------
    opened_mask : ndarray
        Morphologically opened mask
    """
    # Use a disk-shaped structuring element
    # This helps separate vacuoles connected by thin bridges
    footprint = morphology.disk(max(1, opening_disk_radius))
    opened = morphology.binary_opening(binary_mask, footprint=footprint)
    
    # Erosion followed by dilation (opening) may disconnect some vacuoles
    # but it can also remove small vacuoles entirely
    # So we combine the opened result with the original to recover small objects
    
    # Get small objects that were removed by opening
    removed = binary_mask & ~opened
    small_objects, num = ndimage.label(removed)
    
    # Add back small objects that meet size criteria
    # Only recover objects at least as large as the structuring element
    min_recoverable_size = np.pi * opening_disk_radius ** 2
    for i in range(1, num + 1):
        obj_mask = small_objects == i
        obj_size = np.sum(obj_mask)
        if obj_size >= min_recoverable_size:
            opened |= obj_mask
    
    return opened


def multi_threshold_segmentation(vacuole_img, sigma=1.3488):
    """
    Apply multiple thresholding methods and combine results.
    
    This helps capture vacuoles with varying intensities.
    
    Parameters
    ----------
    vacuole_img : ndarray
        Vacuole channel image
    sigma : float
        Gaussian smoothing parameter
    
    Returns
    -------
    combined_mask : ndarray
        Combined binary mask from multiple thresholds
    """
    # Clip and transform
    vacuole_img = np.clip(vacuole_img, a_min=0, a_max=None)
    vacuole_log = exposure.adjust_log(vacuole_img + 1)  # +1 to avoid log(0)
    vacuole_smooth = filters.gaussian(vacuole_log, sigma=sigma)
    
    # Method 1: Otsu (original method)
    thresh_otsu = filters.threshold_otsu(vacuole_smooth)
    mask_otsu = vacuole_smooth > thresh_otsu
    
    # Method 2: Li's minimum cross entropy (good for dim objects)
    try:
        thresh_li = filters.threshold_li(vacuole_smooth)
        mask_li = vacuole_smooth > thresh_li
    except:
        mask_li = mask_otsu
    
    # Method 3: Yen's method (good for high dynamic range)
    try:
        thresh_yen = filters.threshold_yen(vacuole_smooth)
        mask_yen = vacuole_smooth > thresh_yen
    except:
        mask_yen = mask_otsu
    
    # Combine masks using voting (at least 2 out of 3 methods agree)
    vote_sum = mask_otsu.astype(int) + mask_li.astype(int) + mask_yen.astype(int)
    combined_mask = vote_sum >= 2
    
    # Fill holes
    combined_mask = ndimage.binary_fill_holes(combined_mask)
    
    return combined_mask


def segment_vacuoles_improved(
    image,
    vacuole_channel_index,
    nuclei_channel_index=None,
    cell_masks=None,
    cytoplasm_masks=None,
    vacuole_min_size=10,
    vacuole_max_size=200,
    threshold_smoothing_scale=1.3488,
    min_distance_between_maxima=20,
    max_objects_per_cell=120,
    overlap_threshold=0.1,
    nuclei_min_distance=5,
    nuclei_centroids=None,
    nuclei_detection=False,
    use_multi_threshold=True,
    use_morphological_opening=True,
    use_enhanced_declumping=True,
    h_minima_percentile=20,
    opening_disk_radius=1,
):
    """
    Improved vacuole segmentation with better clump handling.
    
    New parameters
    --------------
    use_multi_threshold : bool, optional
        Use multiple thresholding methods combined (default True)
    use_morphological_opening : bool, optional
        Apply morphological opening to separate connected vacuoles (default True)
    use_enhanced_declumping : bool, optional
        Use enhanced watershed with h-minima transform (default True)
    h_minima_percentile : float, optional
        Percentile for auto-calculating h-minima threshold (default 20)
        Lower values = more aggressive suppression of shallow peaks
    opening_disk_radius : int, optional
        Radius of disk structuring element for morphological opening (default 1)
        Controls how thick bridges need to be to get cut:
        - radius=1: Cuts bridges 1-2 pixels wide (very thin, default)
        - radius=2: Cuts bridges 2-4 pixels wide (thin)
        - radius=3: Cuts bridges 3-6 pixels wide (medium)
        - radius=4: Cuts bridges 4-8 pixels wide (thick)
        Larger radius = more aggressive bridge cutting but may remove small vacuoles
    
    All other parameters are the same as the original function.
    """
    segment_cells = cell_masks is not None
    
    # Extract and prepare vacuole channel
    vacuole_img = image[vacuole_channel_index]
    vacuole_img = np.clip(vacuole_img, a_min=0, a_max=None)
    
    # Apply thresholding (improved method)
    if use_multi_threshold:
        print("Using multi-threshold segmentation...")
        binary_mask = multi_threshold_segmentation(vacuole_img, sigma=threshold_smoothing_scale)
    else:
        # Original method
        vacuole_log = exposure.adjust_log(vacuole_img + 1)
        vacuole_smooth = filters.gaussian(vacuole_log, sigma=threshold_smoothing_scale)
        thresh = filters.threshold_otsu(vacuole_smooth)
        binary_mask = vacuole_smooth > thresh
        binary_mask = ndimage.binary_fill_holes(binary_mask)
    
    # Check if we found anything
    if not np.any(binary_mask):
        print("No objects detected after thresholding")
        return create_empty_results(
            cell_masks, cytoplasm_masks, nuclei_detection, nuclei_centroids, segment_cells
        )
    
    print(f"Initial mask: {np.sum(binary_mask)} pixels")
    
    # Apply morphological opening to separate connected vacuoles
    if use_morphological_opening:
        print("Applying morphological opening...")
        binary_mask = apply_morphological_opening(binary_mask, opening_disk_radius=opening_disk_radius)
        print(f"After opening: {np.sum(binary_mask)} pixels")
    
    # Declumping with enhanced watershed
    if use_enhanced_declumping:
        print("Applying enhanced declumping...")
        h_minima = None  # Will be auto-calculated based on percentile
        declumped = enhanced_declumping(
            binary_mask, 
            vacuole_img, 
            min_distance=min_distance_between_maxima,
            h_minima=h_minima
        )
    else:
        # Original watershed method
        print("Applying standard watershed...")
        distance = ndimage.distance_transform_edt(binary_mask)
        local_max = feature.peak_local_max(
            distance, min_distance=min_distance_between_maxima, labels=binary_mask
        )
        markers = np.zeros_like(binary_mask, dtype=int)
        if len(local_max) > 0:
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
            declumped = segmentation.watershed(-distance, markers, mask=binary_mask)
            missing = (declumped == 0) & binary_mask
            if np.any(missing):
                labeled_missing, _ = ndimage.label(missing)
                labeled_missing[labeled_missing > 0] += declumped.max()
                declumped += labeled_missing
        else:
            declumped, _ = ndimage.label(binary_mask)
    
    print(f"After declumping: {len(np.unique(declumped)) - 1} objects")
    
    # Fill holes after declumping
    unique_labels = np.unique(declumped[declumped > 0])
    for label in unique_labels:
        mask = declumped == label
        filled = ndimage.binary_fill_holes(mask)
        declumped[filled] = label
    
    # Filter by diameter using Feret diameters
    print("Filtering by diameter...")
    regions = measure.regionprops(declumped)
    valid_labels = []
    
    for region in regions:
        coords = region.coords[:, [1, 0]]  # Convert to (x, y)
        if len(coords) < 3:
            continue
        
        feret_min, feret_max = get_feret_diameters(coords)
        
        if vacuole_min_size <= feret_min and feret_max <= vacuole_max_size:
            valid_labels.append(region.label)
    
    if not valid_labels:
        print("No valid vacuoles found after diameter filtering")
        return create_empty_results(
            cell_masks, cytoplasm_masks, nuclei_detection, nuclei_centroids, segment_cells
        )
    
    print(f"After diameter filtering: {len(valid_labels)} valid vacuoles")
    
    # Create valid vacuoles mask with renumbered labels
    labeled_vacuoles = np.zeros_like(declumped)
    for i, lbl in enumerate(valid_labels, start=1):
        labeled_vacuoles[declumped == lbl] = i
    
    num_vacuoles = len(valid_labels)
    
    # Get cell IDs
    if segment_cells:
        cell_ids = np.unique(cell_masks[cell_masks > 0])
    else:
        cell_ids = np.array([])
    
    # Prepare nuclei detection if enabled
    nuclei_img = None
    if nuclei_detection:
        nuclei_channel_index = (
            vacuole_channel_index if nuclei_channel_index is None else nuclei_channel_index
        )
        nuclei_img = image[nuclei_channel_index]
    
    # Prepare nuclei centroids
    nuclei_centroids_dict = None
    if nuclei_centroids is not None:
        if isinstance(nuclei_centroids, pd.DataFrame):
            nuclei_centroids_dict = {
                row.get("nuclei_id", idx): (row["i"], row["j"])
                for idx, row in nuclei_centroids.iterrows()
            }
        else:
            nuclei_centroids_dict = nuclei_centroids
    
    # Pre-compute region properties
    vacuole_regions = {
        region.label: region for region in measure.regionprops(labeled_vacuoles)
    }
    
    # Initialize tracking
    vacuole_cell_mapping = []
    if segment_cells:
        vacuoles_per_cell = {cell_id: 0 for cell_id in cell_ids}
    else:
        vacuoles_per_cell = {}
    
    # Process each vacuole (same as original)
    print("Processing vacuole-cell associations...")
    for vacuole_id in range(1, num_vacuoles + 1):
        if vacuole_id not in vacuole_regions:
            continue
        
        region = vacuole_regions[vacuole_id]
        vacuole_mask = labeled_vacuoles == vacuole_id
        vacuole_area = region.area
        vacuole_centroid = region.centroid
        vacuole_diameter = 2 * np.sqrt(vacuole_area / np.pi)
        
        mapping_entry = {
            "vacuole_id": vacuole_id,
            "vacuole_area": vacuole_area,
            "vacuole_diameter": vacuole_diameter,
        }
        
        # Nuclei detection within vacuoles
        if nuclei_detection and nuclei_img is not None:
            peaks = feature.peak_local_max(
                nuclei_img,
                min_distance=nuclei_min_distance,
                labels=vacuole_mask,
                exclude_border=False,
            )
            mapping_entry["nuclei_count"] = len(peaks)
            mapping_entry["peak_coordinates"] = peaks.tolist() if len(peaks) > 0 else []
        elif nuclei_detection:
            mapping_entry["nuclei_count"] = 0
            mapping_entry["peak_coordinates"] = []
        
        # Distance to nearest cell nucleus
        if nuclei_centroids_dict is not None:
            min_dist = np.inf
            nearest_nucleus_id = None
            for nuc_id, nuc_centroid in nuclei_centroids_dict.items():
                dist = np.sqrt(
                    (vacuole_centroid[0] - nuc_centroid[0]) ** 2
                    + (vacuole_centroid[1] - nuc_centroid[1]) ** 2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_nucleus_id = nuc_id
            
            mapping_entry["distance_to_nucleus"] = min_dist if min_dist != np.inf else None
            mapping_entry["nearest_nucleus_id"] = nearest_nucleus_id
        
        # Cell association
        if segment_cells:
            best_cell_id = None
            best_overlap = 0
            
            for cell_id in cell_ids:
                if vacuoles_per_cell[cell_id] >= max_objects_per_cell:
                    continue
                
                cell_mask = cell_masks == cell_id
                overlap = np.sum(vacuole_mask & cell_mask)
                
                if overlap > 0:
                    overlap_ratio = overlap / vacuole_area
                    if overlap_ratio >= overlap_threshold and overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_cell_id = cell_id
            
            if best_cell_id is not None:
                mapping_entry["cell_id"] = best_cell_id
                mapping_entry["overlap_ratio"] = best_overlap
                vacuole_cell_mapping.append(mapping_entry)
                vacuoles_per_cell[best_cell_id] += 1
        else:
            mapping_entry["cell_id"] = None
            mapping_entry["overlap_ratio"] = None
            vacuole_cell_mapping.append(mapping_entry)
    
    # Create DataFrames and summaries (same as original)
    vacuole_cell_df = pd.DataFrame(vacuole_cell_mapping)
    
    # Create cell summary
    if segment_cells and vacuole_cell_mapping:
        grouped = vacuole_cell_df.groupby("cell_id")
        cell_summary = []
        
        for cell_id in cell_ids:
            cell_area = np.sum(cell_masks == cell_id)
            summary_entry = {"cell_id": cell_id, "cell_area": cell_area}
            
            if cell_id in grouped.groups:
                cell_vacuoles = grouped.get_group(cell_id)
                total_vacuole_area = cell_vacuoles["vacuole_area"].sum()
                mean_diameter = cell_vacuoles["vacuole_diameter"].mean()
                
                summary_entry.update({
                    "has_vacuole": True,
                    "num_vacuoles": len(cell_vacuoles),
                    "vacuole_ids": list(cell_vacuoles["vacuole_id"]),
                    "total_vacuole_area": total_vacuole_area,
                    "vacuole_area_ratio": total_vacuole_area / cell_area if cell_area > 0 else 0,
                    "mean_vacuole_diameter": mean_diameter,
                })
                
                if nuclei_detection:
                    summary_entry.update({
                        "total_nuclei_in_vacuoles": cell_vacuoles["nuclei_count"].sum(),
                        "multinucleated_vacuole_count": len(cell_vacuoles[cell_vacuoles["nuclei_count"] > 1]),
                    })
                
                if nuclei_centroids_dict is not None:
                    mean_distance = (
                        cell_vacuoles["distance_to_nucleus"].dropna().mean()
                        if not cell_vacuoles["distance_to_nucleus"].dropna().empty
                        else None
                    )
                    summary_entry["mean_distance_to_nucleus"] = mean_distance
            else:
                summary_entry.update({
                    "has_vacuole": False,
                    "num_vacuoles": 0,
                    "vacuole_ids": [],
                    "total_vacuole_area": 0,
                    "vacuole_area_ratio": 0,
                    "mean_vacuole_diameter": None,
                })
                
                if nuclei_detection:
                    summary_entry.update({
                        "total_nuclei_in_vacuoles": 0,
                        "multinucleated_vacuole_count": 0,
                    })
                
                if nuclei_centroids_dict is not None:
                    summary_entry["mean_distance_to_nucleus"] = None
            
            cell_summary.append(summary_entry)
    
    elif segment_cells and not vacuole_cell_mapping:
        cell_summary = []
        for cell_id in cell_ids:
            cell_area = np.sum(cell_masks == cell_id)
            summary_entry = {
                "cell_id": cell_id,
                "has_vacuole": False,
                "num_vacuoles": 0,
                "vacuole_ids": [],
                "cell_area": cell_area,
                "total_vacuole_area": 0,
                "vacuole_area_ratio": 0,
                "mean_vacuole_diameter": None,
            }
            
            if nuclei_detection:
                summary_entry.update({
                    "total_nuclei_in_vacuoles": 0,
                    "multinucleated_vacuole_count": 0,
                })
            
            if nuclei_centroids_dict is not None:
                summary_entry["mean_distance_to_nucleus"] = None
            
            cell_summary.append(summary_entry)
    
    else:
        # No cell segmentation
        cell_summary = [{
            "cell_id": None,
            "has_vacuole": len(vacuole_cell_mapping) > 0,
            "num_vacuoles": len(vacuole_cell_mapping),
            "vacuole_ids": [v["vacuole_id"] for v in vacuole_cell_mapping],
            "cell_area": None,
            "total_vacuole_area": sum(v["vacuole_area"] for v in vacuole_cell_mapping) if vacuole_cell_mapping else 0,
            "vacuole_area_ratio": None,
            "mean_vacuole_diameter": vacuole_cell_df["vacuole_diameter"].mean() if len(vacuole_cell_df) > 0 else None,
        }]
        
        if nuclei_detection and len(vacuole_cell_df) > 0:
            cell_summary[0].update({
                "total_nuclei_in_vacuoles": vacuole_cell_df["nuclei_count"].sum(),
                "multinucleated_vacuole_count": len(vacuole_cell_df[vacuole_cell_df["nuclei_count"] > 1]),
            })
        elif nuclei_detection:
            cell_summary[0].update({
                "total_nuclei_in_vacuoles": 0,
                "multinucleated_vacuole_count": 0,
            })
        
        if nuclei_centroids_dict is not None:
            if len(vacuole_cell_df) > 0:
                cell_summary[0]["mean_distance_to_nucleus"] = (
                    vacuole_cell_df["distance_to_nucleus"].dropna().mean()
                    if not vacuole_cell_df["distance_to_nucleus"].dropna().empty
                    else None
                )
            else:
                cell_summary[0]["mean_distance_to_nucleus"] = None
    
    # Create results
    cell_summary_df = pd.DataFrame(cell_summary)
    cell_vacuole_table = {
        "cell_summary": cell_summary_df,
        "vacuole_cell_mapping": vacuole_cell_df,
    }
    
    # Create associated vacuoles mask
    associated_vacuoles = np.zeros_like(labeled_vacuoles)
    for mapping in vacuole_cell_mapping:
        vacuole_id = mapping["vacuole_id"]
        vacuole_mask = labeled_vacuoles == vacuole_id
        associated_vacuoles[vacuole_mask] = vacuole_id
    
    # Print statistics
    total_kept = len(vacuole_cell_mapping)
    if segment_cells:
        print(f"Kept {total_kept} out of {num_vacuoles} detected vacuoles ({total_kept / num_vacuoles * 100:.1f}%)")
        print(f"Discarded {num_vacuoles - total_kept} vacuoles that didn't meet criteria")
    else:
        print(f"Kept all {total_kept} detected vacuoles (no cell association required)")
    
    # Update cytoplasm masks
    updated_cytoplasm_masks = None
    if cytoplasm_masks is not None:
        updated_cytoplasm_masks = cytoplasm_masks.copy()
        for mapping in vacuole_cell_mapping:
            vacuole_id = mapping["vacuole_id"]
            cell_id = mapping["cell_id"]
            vacuole_mask = associated_vacuoles == vacuole_id
            if cell_id is not None:
                cytoplasm_mask = updated_cytoplasm_masks == cell_id
                updated_cytoplasm_masks[cytoplasm_mask & vacuole_mask] = 0
        print(f"Updated cytoplasm masks by removing {len(vacuole_cell_mapping)} vacuole regions")
    
    # Return results
    if updated_cytoplasm_masks is not None:
        return associated_vacuoles, cell_vacuole_table, updated_cytoplasm_masks
    else:
        return associated_vacuoles, cell_vacuole_table


def create_empty_results(
    cell_masks, 
    cytoplasm_masks, 
    nuclei_detection=False, 
    nuclei_centroids=None,
    segment_cells=True
):
    """Helper function to create empty results when no vacuoles are found."""
    
    if segment_cells and cell_masks is not None:
        cell_ids = np.unique(cell_masks[cell_masks > 0])
        empty_vacuole_masks = np.zeros_like(cell_masks)
        
        cell_summary = []
        for cell_id in cell_ids:
            cell_area = np.sum(cell_masks == cell_id)
            summary_entry = {
                "cell_id": cell_id,
                "has_vacuole": False,
                "num_vacuoles": 0,
                "vacuole_ids": [],
                "cell_area": cell_area,
                "total_vacuole_area": 0,
                "vacuole_area_ratio": 0,
                "mean_vacuole_diameter": None,
            }
            
            if nuclei_detection:
                summary_entry.update({
                    "total_nuclei_in_vacuoles": 0,
                    "multinucleated_vacuole_count": 0,
                })
            
            if nuclei_centroids is not None:
                summary_entry["mean_distance_to_nucleus"] = None
            
            cell_summary.append(summary_entry)
    
    else:
        if cell_masks is not None:
            empty_vacuole_masks = np.zeros_like(cell_masks)
        elif cytoplasm_masks is not None:
            empty_vacuole_masks = np.zeros_like(cytoplasm_masks)
        else:
            raise ValueError("Need either cell_masks or cytoplasm_masks to determine image shape")
        
        cell_summary = [{
            "cell_id": None,
            "has_vacuole": False,
            "num_vacuoles": 0,
            "vacuole_ids": [],
            "cell_area": None,
            "total_vacuole_area": 0,
            "vacuole_area_ratio": None,
            "mean_vacuole_diameter": None,
        }]
        
        if nuclei_detection:
            cell_summary[0].update({
                "total_nuclei_in_vacuoles": 0,
                "multinucleated_vacuole_count": 0,
            })
        
        if nuclei_centroids is not None:
            cell_summary[0]["mean_distance_to_nucleus"] = None
    
    cell_vacuole_table = {
        "cell_summary": pd.DataFrame(cell_summary),
        "vacuole_cell_mapping": pd.DataFrame(),
    }
    
    if cytoplasm_masks is not None:
        return empty_vacuole_masks, cell_vacuole_table, cytoplasm_masks
    else:
        return empty_vacuole_masks, cell_vacuole_table