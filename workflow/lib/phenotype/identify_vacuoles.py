"""Improved vacuole segmentation with enhanced declumping strategies.

Key improvements for handling clumping:
1. Multi-scale approach with adaptive thresholding
2. Enhanced watershed with multiple marker strategies
3. Morphological operations for better separation
4. H-minima transform to suppress spurious local maxima
5. Shape-based declumping with boundary proportion filtering
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
    Enhanced declumping using distance transform and intensity-based markers.
    
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
        If None, automatically calculated as 20th percentile of distance values
    
    Returns
    -------
    declumped : ndarray
        Labeled segmentation mask
    """
    # Distance transform with adaptive h-minima
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Auto-calculate h_minima if not provided (suppress bottom 20% of peaks)
    if h_minima is None:
        h_minima = np.percentile(distance[distance > 0], 20)
    
    # Apply h-minima transform to suppress spurious local maxima
    distance_filtered = morphology.h_minima(distance, h_minima)
    
    # Find distance-based peaks
    local_max = feature.peak_local_max(
        distance_filtered,
        min_distance=min_distance,
        labels=binary_mask,
        exclude_border=True
    )
    
    # Find intensity-based peaks and merge with distance peaks
    if vacuole_img is not None:
        smoothed_intensity = filters.gaussian(vacuole_img, sigma=2)
        intensity_peaks = feature.peak_local_max(
            smoothed_intensity,
            min_distance=min_distance // 2,
            labels=binary_mask,
            exclude_border=True,
            threshold_rel=0.3
        )
        
        # Merge peaks, avoiding duplicates
        all_peaks = [tuple(peak) for peak in local_max]
        for peak in intensity_peaks:
            is_far_enough = all(
                np.sqrt((peak[0] - ep[0])**2 + (peak[1] - ep[1])**2) >= min_distance / 2
                for ep in all_peaks
            )
            if is_far_enough:
                all_peaks.append(tuple(peak))
        
        markers_array = np.array(all_peaks)
    else:
        markers_array = local_max
    
    # Create markers and apply watershed
    markers = np.zeros_like(binary_mask, dtype=int)
    if len(markers_array) > 0:
        markers[tuple(markers_array.T)] = np.arange(1, len(markers_array) + 1)
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
        Radius of disk structuring element (larger = more aggressive)
    
    Returns
    -------
    opened_mask : ndarray
        Morphologically opened mask
    """
    footprint = morphology.disk(max(1, opening_disk_radius))
    opened = morphology.binary_opening(binary_mask, footprint=footprint)
    
    # Recover small objects that were removed by opening
    removed = binary_mask & ~opened
    small_objects, num = ndimage.label(removed)
    
    # Only recover objects at least as large as the structuring element
    min_recoverable_size = np.pi * opening_disk_radius ** 2
    for i in range(1, num + 1):
        obj_mask = small_objects == i
        if np.sum(obj_mask) >= min_recoverable_size:
            opened |= obj_mask
    
    return opened


def multi_threshold_segmentation(vacuole_img, sigma=1.3488):
    """
    Apply multiple thresholding methods and combine results using majority voting.
    
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
    # Preprocess image
    vacuole_img = np.clip(vacuole_img, a_min=0, a_max=None)
    vacuole_log = exposure.adjust_log(vacuole_img + 1)
    vacuole_smooth = filters.gaussian(vacuole_log, sigma=sigma)
    
    # Apply three thresholding methods
    thresh_otsu = filters.threshold_otsu(vacuole_smooth)
    mask_otsu = vacuole_smooth > thresh_otsu
    
    try:
        thresh_li = filters.threshold_li(vacuole_smooth)
        mask_li = vacuole_smooth > thresh_li
    except:
        mask_li = mask_otsu
    
    try:
        thresh_yen = filters.threshold_yen(vacuole_smooth)
        mask_yen = vacuole_smooth > thresh_yen
    except:
        mask_yen = mask_otsu
    
    # Majority voting (at least 2 out of 3 methods agree)
    vote_sum = mask_otsu.astype(int) + mask_li.astype(int) + mask_yen.astype(int)
    combined_mask = vote_sum >= 2
    
    # Fill holes
    combined_mask = ndimage.binary_fill_holes(combined_mask)
    
    return combined_mask


def shape_based_declumping(binary_mask, vacuole_img=None, min_distance=20, proportion_threshold=0.12):
    """
    Split connected components only when the separating boundary is short
    relative to the region perimeter.

    Parameters
    ----------
    binary_mask : ndarray
        Input binary vacuole mask
    vacuole_img : ndarray, optional
        Intensity image (currently unused, kept for API compatibility)
    min_distance : int
        Minimum distance between peaks for watershed markers
    proportion_threshold : float
        If boundary_length / perimeter < proportion_threshold, accept the split
        Example: 0.12 means cut must be < 12% of perimeter to split

    Returns
    -------
    labeled : ndarray
        Labeled mask after shape-based declumping
    """
    labeled_out = np.zeros_like(binary_mask, dtype=int)
    next_label = 1

    # Label connected regions
    regions_lab, n = ndimage.label(binary_mask)
    
    for region_label in range(1, n + 1):
        region_mask = regions_lab == region_label
        if region_mask.sum() == 0:
            continue

        # Distance transform and find peaks
        dist = ndimage.distance_transform_edt(region_mask)
        peaks = feature.peak_local_max(
            dist, 
            min_distance=min_distance, 
            labels=region_mask, 
            exclude_border=False
        )
        
        # If only one peak, keep as single object
        if len(peaks) <= 1:
            labeled_out[region_mask] = next_label
            next_label += 1
            continue

        # Create markers and apply watershed
        markers = np.zeros_like(region_mask, dtype=int)
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
        local_watershed = segmentation.watershed(-dist, markers, mask=region_mask)

        # Compute boundary length between watershed regions
        boundary_mask = np.zeros_like(region_mask, dtype=bool)
        lab = local_watershed
        
        for dy, dx in ((0, 1), (1, 0), (-1, 0), (0, -1)):
            neighbor = np.roll(lab, shift=(dy, dx), axis=(0, 1))
            # Zero out rolled-in edges
            if dy == 1:
                neighbor[0, :] = 0
            elif dy == -1:
                neighbor[-1, :] = 0
            if dx == 1:
                neighbor[:, 0] = 0
            elif dx == -1:
                neighbor[:, -1] = 0
            
            boundary_mask |= (lab != neighbor) & (lab > 0) & (neighbor > 0)

        boundary_length = np.sum(boundary_mask)
        prop = measure.regionprops(region_mask.astype(np.uint8))[0]
        perimeter = prop.perimeter if prop.perimeter > 0 else 1.0

        # Accept split if boundary is short relative to perimeter
        if (boundary_length / perimeter) < proportion_threshold:
            sublabels = np.unique(local_watershed[local_watershed > 0])
            for s in sublabels:
                labeled_out[local_watershed == s] = next_label
                next_label += 1
        else:
            # Reject split, keep as single object
            labeled_out[region_mask] = next_label
            next_label += 1

    return labeled_out


def segment_vacuoles_improved(
    image,
    vacuole_channel_index,
    nuclei_channel_index=None,
    cell_masks=None,
    cytoplasm_masks=None,
    vacuole_min_size=10,
    vacuole_max_size=200,
    threshold_smoothing_scale=0,
    min_distance_between_maxima=20,
    max_objects_per_cell=120,
    max_total_objects=300,
    overlap_threshold=0.1,
    nuclei_min_distance=5,
    nuclei_centroids=None,
    nuclei_detection=False,
    use_multi_threshold=False,
    use_morphological_opening=True,
    use_enhanced_declumping=True,
    opening_disk_radius=1,
    use_shape_based_declumping=True,
    proportion_threshold=0.4,
):
    """
    Improved vacuole segmentation with better clump handling.
    
    Parameters
    ----------
    image : ndarray
        Multi-channel image
    vacuole_channel_index : int
        Index of vacuole channel
    nuclei_channel_index : int, optional
        Index of nuclei channel for detection
    cell_masks : ndarray, optional
        Cell segmentation masks
    cytoplasm_masks : ndarray, optional
        Cytoplasm segmentation masks
    vacuole_min_size : float
        Minimum Feret diameter for valid vacuoles
    vacuole_max_size : float
        Maximum Feret diameter for valid vacuoles
    threshold_smoothing_scale : float
        Gaussian smoothing sigma for thresholding
    min_distance_between_maxima : int
        Minimum distance between watershed markers
    max_objects_per_cell : int
        Maximum vacuoles to assign per cell
    max_total_objects : int, optional
        **FAILSAFE**: If more than this many objects detected after thresholding,
        return empty results. Use this to prevent processing images with 
        extreme over-segmentation. If None, no limit is applied.
    overlap_threshold : float
        Minimum overlap ratio for vacuole-cell association
    nuclei_min_distance : int
        Minimum distance between nuclei peaks
    nuclei_centroids : dict or DataFrame, optional
        Nuclei centroid positions
    nuclei_detection : bool
        Whether to detect nuclei within vacuoles
    use_multi_threshold : bool
        Use multiple thresholding methods combined
    use_morphological_opening : bool
        Apply morphological opening to separate connected vacuoles
    use_enhanced_declumping : bool
        Use enhanced watershed with h-minima transform
    opening_disk_radius : int
        Radius of disk structuring element for morphological opening
    use_shape_based_declumping : bool
        Apply shape-based declumping refinement
    proportion_threshold : float
        Boundary/perimeter threshold for shape-based splitting
    
    Returns
    -------
    associated_vacuoles : ndarray
        Labeled mask of vacuoles assigned to cells
    cell_vacuole_table : dict
        Dictionary with 'cell_summary' and 'vacuole_cell_mapping' DataFrames
    updated_cytoplasm_masks : ndarray, optional
        Cytoplasm masks with vacuoles removed (if cytoplasm_masks provided)
    """
    segment_cells = cell_masks is not None
    
    vacuole_img = image[vacuole_channel_index]
    vacuole_img = np.clip(vacuole_img, a_min=0, a_max=None)

    # --- Thresholding ---
    if use_multi_threshold:
        binary_mask = multi_threshold_segmentation(vacuole_img, sigma=threshold_smoothing_scale)
    else:
        vacuole_log = exposure.adjust_log(vacuole_img + 1)
        vacuole_smooth = filters.gaussian(vacuole_log, sigma=threshold_smoothing_scale)
        thresh = filters.threshold_otsu(vacuole_smooth)
        binary_mask = vacuole_smooth > thresh
        binary_mask = ndimage.binary_fill_holes(binary_mask)
    
    if not np.any(binary_mask):
        print("No objects detected after thresholding")
        return create_empty_results(cell_masks, cytoplasm_masks, nuclei_detection, nuclei_centroids, segment_cells)

    # --- FAILSAFE: Check for excessive objects early ---
    if max_total_objects is not None:
        # Quick check: count connected components before expensive processing
        temp_labeled, num_components = ndimage.label(binary_mask)
        if num_components > max_total_objects:
            print(f"FAILSAFE TRIGGERED: Detected {num_components} objects (limit: {max_total_objects})")
            print("Returning zero masks to avoid processing over-segmented image")
            return create_empty_results(cell_masks, cytoplasm_masks, nuclei_detection, nuclei_centroids, segment_cells)

    # --- Morphological opening ---
    if use_morphological_opening:
        binary_mask = apply_morphological_opening(binary_mask, opening_disk_radius=opening_disk_radius)

    # --- Enhanced declumping ---
    if use_enhanced_declumping:
        declumped = enhanced_declumping(
            binary_mask,
            vacuole_img,
            min_distance=min_distance_between_maxima,
            h_minima=None
        )
    else:
        distance = ndimage.distance_transform_edt(binary_mask)
        local_max = feature.peak_local_max(distance, min_distance=min_distance_between_maxima, labels=binary_mask)
        markers = np.zeros_like(binary_mask, dtype=int)
        if len(local_max) > 0:
            markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
            declumped = segmentation.watershed(-distance, markers, mask=binary_mask)
        else:
            declumped, _ = ndimage.label(binary_mask)

    print(f"After enhanced declumping: {len(np.unique(declumped)) - 1} objects")

    # --- Shape-based declumping ---
    if use_shape_based_declumping:
        print("Applying shape-based declumping refinement...")
        declumped = shape_based_declumping(
            declumped > 0,
            vacuole_img=vacuole_img,
            min_distance=min_distance_between_maxima,
            proportion_threshold=proportion_threshold,
        )
        print(f"After shape-based declumping: {len(np.unique(declumped)) - 1} objects")
    
    # Fill holes after declumping
    unique_labels = np.unique(declumped[declumped > 0])
    for label in unique_labels:
        mask = declumped == label
        filled = ndimage.binary_fill_holes(mask)
        declumped[filled] = label
    
    # --- Filter by diameter using Feret diameters ---
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
        return create_empty_results(cell_masks, cytoplasm_masks, nuclei_detection, nuclei_centroids, segment_cells)
    
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
    
    # Prepare nuclei detection
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
    
    # --- Process each vacuole ---
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
    
    # --- Create DataFrames and summaries ---
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