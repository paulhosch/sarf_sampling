import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from ..io.points import save_points_df
from config import PROJECTION_METERS, WATER_BAND_NAME, RANDOM_SEED, LABEL_BAND_NAME
from ..utils.stats import update_sampling_stats

def random_sampling(
    path_to_points: str, 
    n_samples: int, 
    strata_distribution: str,
    output_dir: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Perform random sampling on points from a GeoDataFrame or file.
    
    Parameters:
    -----------
    path_to_points : str
        Path to the input points file
    n_samples : int
        Number of samples to draw
    strata_distribution : str
        How to distribute samples between strata. Must be one of:
        - 'simple': Simple random sampling without stratification
        - 'proportional': Samples distributed proportionally to stratum size
        - 'balanced': Equal number of samples per stratum
    output_dir : str or Path
        Path to save the sampled points
        
    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame containing the sampled points
    """
    # Convert output_dir to Path if provided
    if output_dir is not None:
        output_dir = Path(output_dir)
    
    # Load from GeoJSON format
    gdf = gpd.read_file(path_to_points)
    gdf = gdf.to_crs(PROJECTION_METERS)
    
    # Check for valid strata_distribution
    if strata_distribution not in ['simple', 'proportional', 'balanced']:
        raise ValueError("Invalid strata_distribution. Must be 'simple', 'proportional', or 'balanced'")
    
    # Simple random sampling without stratification
    if strata_distribution == 'simple':
        # Ensure n_samples is not larger than the number of available points
        n_samples = min(n_samples, len(gdf))
        
        # Simple random sampling from the entire dataset
        sampled_gdf = gdf.sample(n=n_samples, random_state=RANDOM_SEED).copy()
        
        print(f"Selected {len(sampled_gdf)} samples using simple random sampling")
    
    # Stratified sampling (proportional or balanced)
    else:
        # Separate points by strata
        stratum_0 = gdf[gdf[LABEL_BAND_NAME] == 0]
        stratum_1 = gdf[gdf[LABEL_BAND_NAME] == 1]
        
        # Print stratum sizes for debugging
        print(f"Number of non-flooded points (stratum 0): {len(stratum_0)}")
        print(f"Number of flooded points (stratum 1): {len(stratum_1)}")
        
        # Calculate number of samples per stratum
        if strata_distribution == 'proportional':
            # Calculate proportional distribution
            n_samples_0 = int(n_samples * (len(stratum_0) / len(gdf)))
            n_samples_1 = n_samples - n_samples_0
            print(f"Number of proportional non-flooded samples: {n_samples_0}")
            print(f"Number of proportional flooded samples: {n_samples_1}")
        else:  # balanced
            # Equal distribution between strata
            n_samples_0 = n_samples // 2
            n_samples_1 = n_samples - n_samples_0
            print(f"Number of balanced non-flooded samples: {n_samples_0}")
            print(f"Number of balanced flooded samples: {n_samples_1}")
        
        # Ensure we don't sample more than available
        n_samples_0 = min(n_samples_0, len(stratum_0))
        n_samples_1 = min(n_samples_1, len(stratum_1))
        
        # Sample from each stratum
        samples_0 = stratum_0.sample(n=n_samples_0, random_state=RANDOM_SEED) if n_samples_0 > 0 else None
        samples_1 = stratum_1.sample(n=n_samples_1, random_state=RANDOM_SEED) if n_samples_1 > 0 else None
        
        # Combine the samples
        samples = []
        if samples_0 is not None and len(samples_0) > 0:
            samples.append(samples_0)
        if samples_1 is not None and len(samples_1) > 0:
            samples.append(samples_1)
            
        if not samples:
            raise ValueError("No samples could be drawn from any stratum")
            
        sampled_gdf = pd.concat(samples).copy()
        
        print(f"Selected {len(sampled_gdf)} samples using {strata_distribution} random sampling")
        print(f"Number of samples per stratum: {sampled_gdf[LABEL_BAND_NAME].value_counts().to_dict()}")
    
    # Add a sample_id column if not present
    if 'sample_id' not in sampled_gdf.columns:
        sampled_gdf['sample_id'] = range(len(sampled_gdf))
    
    # Save if output_dir is provided
    if output_dir is not None:
        output_filename = f"{strata_distribution}_random.geojson"
        save_points_df(sampled_gdf, output_dir / output_filename, overwrite=True)
        print(f"Saved samples to {output_dir / output_filename}")
        update_sampling_stats(
            output_dir=output_dir,
            sampling_method=f"random",
            strata_distribution=strata_distribution,
            sampled_gdf=sampled_gdf,
            output_filename=output_filename,
            label_column=LABEL_BAND_NAME
        )
    
    return sampled_gdf 