import geopandas as gpd
from pathlib import Path
from typing import Optional, Union
from ..io.points import save_points_df

def simple_random_sample(
    path_to_points: str, 
    samples_per_stratum: dict, 
    output_path: Optional[Union[str, Path]] = None,
    random_state: Optional[int] = None
) -> gpd.GeoDataFrame:
    """
    Perform simple random sampling on points from a GeoDataFrame or file.
    
    Parameters:
    -----------
    path_to_points : str
        Path to the input points file
    samples_per_stratum : dict
        Dictionary with required samples per stratum {0: n_non_flooded, 1: n_flooded}
        Values will be summed up to determine total number of samples
    output_path : str or Path, optional
        Path to save the sampled points. If None, won't save to file
    random_state : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame containing the sampled points
    """
    # Calculate total number of samples needed
    n_samples = sum(samples_per_stratum.values())
    
    # Load from GeoJSON format
    gdf = gpd.read_file(path_to_points)
    
    # Ensure n_samples is not larger than the number of available points
    n_samples = min(n_samples, len(gdf))
    
    # Use pandas sample instead of geopandas sample_points since we already have points
    sampled_gdf = gdf.sample(n=n_samples, random_state=random_state).copy()
    
    # Add a sample_id column if not present
    if 'sample_id' not in sampled_gdf.columns:
        sampled_gdf['sample_id'] = range(len(sampled_gdf))
    
    # Save 
    save_points_df(sampled_gdf, output_path)
    print(f"Selected {len(sampled_gdf)} samples using simple random sampling")
    
    return sampled_gdf 