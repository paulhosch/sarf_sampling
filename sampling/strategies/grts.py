from config import LABEL_BAND_NAME, PROJECTION_METERS, RANDOM_SEED
import pygrts
import geopandas as gpd
import pandas as pd
import numpy as np

from sampling.io.points import save_points_df
from sampling.processing.grts import get_iterative_grts_sample
from sampling.utils.stats import update_sampling_stats

def simple_grts_sample(points_path, n_samples, output_dir, grid_dir, decrement_factor = 0.5, max_iterations = 5):
    # Load points
    points_gdf = gpd.read_file(points_path)
    points_gdf = points_gdf.to_crs(PROJECTION_METERS)

    # Perform GRTS sampling
    sampled_gdf, grid_length = get_iterative_grts_sample(points_gdf, n_samples, grid_dir / 'simple_grts_grid.geojson', decrement_factor = decrement_factor, max_iterations = max_iterations)
    
    # Save the sample
    output_filename = 'simple_grts_samples.geojson'
    save_points_df(sampled_gdf, output_dir / output_filename, overwrite=True)
    
    # Update stats.csv
    update_sampling_stats(
        output_dir=output_dir,
        sampling_method="simple_grts",
        strata_distribution="simple",
        sampled_gdf=sampled_gdf,
        output_filename=output_filename,
        label_column=LABEL_BAND_NAME,
        simple_grid_length=grid_length
    )
    
    return sampled_gdf

def grts_sampling(points_path, n_samples, strata_distribution, output_dir, grid_dir, decrement_factor = 0.5, max_iterations = 5):
    """
    Perform GRTS sampling on points from a GeoDataFrame or file.
    
    Parameters
    ----------
    points_path : str
        Path to the input points file
    n_samples : int
        Total number of samples to select
    strata_distribution : str
        How to distribute samples between strata. Must be one of:
        - 'simple': Simple random sampling without stratification
        - 'proportional': Samples distributed proportionally to stratum size
        - 'balanced': Equal number of samples per stratum
    output_dir : str
        Path to save the sampled points
    grid_dir : str
        Path to save the grids
        
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the sampled points with strata labels
    """
    # If using simple distribution, call the simple_grts_sample function
    if strata_distribution == 'simple':
        return simple_grts_sample(points_path, n_samples, output_dir, grid_dir, decrement_factor, max_iterations)
    
    # Load points
    points_gdf = gpd.read_file(points_path)
    points_gdf = points_gdf.to_crs(PROJECTION_METERS)

    # Ensure we're using integers for strata values
    points_gdf[LABEL_BAND_NAME] = points_gdf[LABEL_BAND_NAME].astype(int)
    
    # Seperate points into strata
    non_flooded_points = points_gdf[points_gdf[LABEL_BAND_NAME] == 0]
    flooded_points = points_gdf[points_gdf[LABEL_BAND_NAME] == 1]
    
    # Print stratum sizes for debugging
    print(f"Number of non-flooded points: {len(non_flooded_points)}")
    print(f"Number of flooded points: {len(flooded_points)}")
    
    # Check strata_distribution is valid
    if strata_distribution not in ['proportional', 'balanced']:
        raise ValueError("Invalid strata distribution, must be 'simple', 'proportional', or 'balanced'")

    # Calculate number of samples per stratum
    if strata_distribution == 'proportional':
        # If no strata distribution is provided, use class ratio proportional distribution 
        n_non_flooded = int(n_samples * (len(non_flooded_points) / len(points_gdf)))
        n_flooded = n_samples - n_non_flooded   
        print(f"Number of proportional non-flooded samples: {n_non_flooded}")
        print(f"Number of proportional flooded samples: {n_flooded}")

    elif strata_distribution == 'balanced':
        # If strata distribution is balanced, split n_samples equally between strata
        n_non_flooded = int(n_samples / 2)
        n_flooded = n_samples - n_non_flooded   
        print(f"Number of balanced non-flooded samples: {n_non_flooded}")
        print(f"Number of balanced flooded samples: {n_flooded}")
    
    # Initialize lists to hold samples
    samples = []
    
    # Sample non-flooded points if we have any
    if n_non_flooded > 0 and len(non_flooded_points) > 0:
        # Make sure we don't try to sample more points than available
        if n_non_flooded > len(non_flooded_points):
            print(f"Warning: Reducing non-flooded samples from {n_non_flooded} to {len(non_flooded_points)}")
            n_non_flooded = len(non_flooded_points)
            
        try:
            non_flooded_sample, non_flooded_grid_length = get_iterative_grts_sample(
                non_flooded_points, 
                n_non_flooded, 
                grid_output_path = grid_dir / 'non_flooded_grts_grid.geojson',
                decrement_factor = decrement_factor, 
                max_iterations = max_iterations
            )
            samples.append(non_flooded_sample)
        except Exception as e:
            print(f"Error sampling non-flooded points: {e}")
    
    # Sample flooded points if we have any
    if n_flooded > 0 and len(flooded_points) > 0:
        # Make sure we don't try to sample more points than available
        if n_flooded > len(flooded_points):
            print(f"Warning: Reducing flooded samples from {n_flooded} to {len(flooded_points)}")
            n_flooded = len(flooded_points)
            
        try:
            flooded_sample, flooded_grid_length = get_iterative_grts_sample(
                flooded_points, 
                n_flooded, 
                grid_output_path = grid_dir / 'flooded_grts_grid.geojson',
                decrement_factor = decrement_factor, 
                max_iterations = max_iterations
            )
            samples.append(flooded_sample)
        except Exception as e:
            print(f"Error sampling flooded points: {e}")

    # Check if we have any samples
    if not samples:
        raise ValueError("Could not sample any points from either stratum")
        
    # Combine samples
    combined_sample = pd.concat(samples)

    # Print number of samples per stratum
    print(f"Number of total samples: {len(combined_sample)}")
    print(f"Number of samples per stratum: {combined_sample[LABEL_BAND_NAME].value_counts()}")

    # Save the combined sample with overwrite=True
    output_filename = f"{strata_distribution}_grts_samples.geojson"
    save_points_df(combined_sample, output_dir / output_filename, overwrite=True)
    print(f"Saved samples to {output_dir / output_filename}")
    
    # Update stats.csv
    update_sampling_stats(
        output_dir=output_dir,
        sampling_method=f"grts",
        strata_distribution=strata_distribution,
        sampled_gdf=combined_sample,
        output_filename=output_filename,
        label_column=LABEL_BAND_NAME,
        flooded_grid_length=flooded_grid_length,
        non_flooded_grid_length=non_flooded_grid_length
    )
    return combined_sample