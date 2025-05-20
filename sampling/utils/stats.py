import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.spatial import cKDTree

def update_sampling_stats(
    output_dir, 
    sampling_method, 
    strata_distribution,
    sampled_gdf, 
    output_filename,
    label_column="label",
    calculate_spacing=True,
    simple_grid_length=None,
    flooded_grid_length=None,
    non_flooded_grid_length=None,
    sampling_time_seconds=None
):
    """
    Updates a stats.csv file in the output directory with sampling statistics.
    Creates the file if it doesn't exist, or appends to it if it does.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory where the stats.csv file is/will be stored
    sampling_method : str
        Name of the sampling method (e.g., 'random', 'systematic', 'grts')
    strata_distribution : str
        Distribution type (e.g., 'simple', 'proportional', 'balanced')
    sampled_gdf : GeoDataFrame
        The sampled points GeoDataFrame
    output_filename : str
        Filename of the saved GeoJSON file
    label_column : str
        Column name containing the class labels (default: 'label')
    calculate_spacing : bool
        Whether to calculate minimum spacing between points (default: True)
    simple_grid_length : float
        Grid length used for simple sampling (optional)
    flooded_grid_length : float
        Grid length used for flooded stratum (optional)
    non_flooded_grid_length : float
        Grid length used for non-flooded stratum (optional)
    sampling_time_seconds : float, optional
        Time taken for sampling in seconds
    """
    # Ensure output_dir is a Path object
    output_dir = Path(output_dir)
    stats_file = output_dir / "stats.csv"
    
    # Get statistics from the sampled GeoDataFrame
    total_samples = len(sampled_gdf)
    
    # Count by class if label column exists, otherwise set to 0
    if label_column in sampled_gdf.columns:
        counts = sampled_gdf[label_column].value_counts().to_dict()
        non_flooded = counts.get(0, 0)
        flooded = counts.get(1, 0)
    else:
        non_flooded = 0
        flooded = 0
    
    # Calculate minimum spacing between points if requested
    min_spacing = np.nan
    if calculate_spacing and len(sampled_gdf) > 1:
        try:
            # Extract coordinates from geometry
            coords = np.array([(p.x, p.y) for p in sampled_gdf.geometry])
            
            # Build KD-tree for efficient nearest neighbor search
            tree = cKDTree(coords)
            
            # Query for the distance to the nearest neighbor (k=2 because each point is its own nearest neighbor)
            distances, _ = tree.query(coords, k=2)
            
            # The second column contains distances to the actual nearest neighbor
            # Filter out zero distances (same point) and get the minimum distance
            min_spacing = int(np.round(np.min(distances[:, 1])))

            print(f"Minimum spacing between points: {min_spacing} meters")
        except Exception as e:
            print(f"Could not calculate minimum spacing: {str(e)}")
    
    # Create new row data
    new_row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'spatial_distribution': sampling_method,
        'class_distribution': strata_distribution,
        'total_samples': total_samples,
        'non_flooded': non_flooded,
        'flooded': flooded,
        'min_spacing': min_spacing,
        'simple_grid_length': simple_grid_length,
        'flooded_grid_length': flooded_grid_length,
        'non_flooded_grid_length': non_flooded_grid_length,
        'sampling_time_seconds': sampling_time_seconds,
        'file_path': output_filename
    }
    
    # Check if file exists
    if stats_file.exists():
        # Load existing file and append
        stats_df = pd.read_csv(stats_file)
        stats_df = pd.concat([stats_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Create new file with headers
        stats_df = pd.DataFrame([new_row])
    
    # Save the updated stats
    stats_df.to_csv(stats_file, index=False)
    print(f"Updated sampling statistics in {stats_file}")
    
    return stats_file
