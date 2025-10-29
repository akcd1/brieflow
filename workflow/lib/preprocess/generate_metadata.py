import os
import pandas as pd

def create_metadata_csv_files(
    samples_df: pd.DataFrame,
    metadata_dir: str,
    image_type: str = "phenotype",
    default_pixel_size: float = 0.65,
    verbose: bool = False
) -> pd.DataFrame:
    """Create metadata CSV files from samples DataFrame and return metadata_samples_df.
    
    This function:
    1. Groups samples by plate/well/round (or cycle)
    2. Creates one metadata CSV per group
    3. Returns a DataFrame pointing to these CSV files (for metadata_samples_df)
    
    Args:
        samples_df: DataFrame with sample information (must have 'sample_fp' column)
                    Should already contain 'channels', 'height', 'width' if available
        metadata_dir: Directory where metadata CSV files should be saved
        image_type: 'phenotype' or 'sbs' 
        default_pixel_size: Default pixel size in microns (default: 0.65 for 20x objective)
        verbose: Print progress information
        
    Returns:
        DataFrame with 'sample_fp' column pointing to created metadata CSV files
    """
    
    os.makedirs(metadata_dir, exist_ok=True)
    
    # Define grouping columns based on image type
    if image_type == 'phenotype':
        group_columns = ['plate', 'well']
        if 'round' in samples_df.columns:
            group_columns.append('round')
    elif image_type == 'sbs':
        group_columns = ['plate', 'well']
        if 'cycle' in samples_df.columns:
            group_columns.append('cycle')
    else:
        raise ValueError(f"Unknown image_type: {image_type}")
    
    # Group samples and create one metadata CSV per group
    metadata_file_info = []
    
    for group_values, group_df in samples_df.groupby(group_columns):
        if verbose:
            print(f"\nProcessing group: {dict(zip(group_columns, group_values))}")
            print(f"  {len(group_df)} files in this group")
        
        metadata_rows = []
        
        for idx, row in group_df.iterrows():
            # Build metadata row
            metadata = {}
            
            # Copy all columns from original row
            for col in group_columns:
                metadata[col] = row[col]
            
            # Add tile if present
            if 'tile' in row:
                metadata['tile'] = row['tile']
            
            # Use existing metadata from samples_df if available, otherwise None
            metadata.update({
                'filename': row['sample_fp'],
                'channels': row.get('channels', None),
                'height': row.get('height', None),
                'width': row.get('width', None),
                'pixel_size_x': default_pixel_size,
                'pixel_size_y': default_pixel_size,
                'x_pos': row.get('x_pos', None),
                'y_pos': row.get('y_pos', None),
                'z_pos': row.get('z_pos', None),
                'pfs_offset': row.get('pfs_offset', None),
            })
            
            metadata_rows.append(metadata)
        
        # Create metadata DataFrame for this group
        group_metadata_df = pd.DataFrame(metadata_rows)
        
        # Build filename for this group
        if isinstance(group_values, tuple):
            parts = [f"{col}_{val}" for col, val in zip(group_columns, group_values)]
        else:
            parts = [f"{group_columns[0]}_{group_values}"]
        filename = "_".join(parts) + "_metadata.tsv"  
        filepath = os.path.join(metadata_dir, filename)

        # Save to CSV with tab separator
        group_metadata_df.to_csv(filepath, sep="\t", index=False)
        
        if verbose:
            print(f"  Saved metadata to: {filepath}")
        
        # Track this metadata file
        file_info = dict(zip(group_columns, group_values if isinstance(group_values, tuple) else [group_values]))
        file_info['sample_fp'] = filepath
        metadata_file_info.append(file_info)
    
    # Create the metadata_samples_df
    metadata_samples_df = pd.DataFrame(metadata_file_info)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Created metadata_samples_df:")
        print(metadata_samples_df)
        print(f"\nThis should be saved and used for phenotype_metadata_samples_df_fp")
        print(f"Total metadata CSV files created: {len(metadata_samples_df)}")
    
    return metadata_samples_df