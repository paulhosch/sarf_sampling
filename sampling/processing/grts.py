import pygrts
import numpy as np
import pandas as pd

from sampling.io.grid import save_grts_grid
from config import RANDOM_SEED


def compute_initial_length(points_gdf, n_samples):
    # Get bounding box of the geometry
    bbox = points_gdf.total_bounds  # [minx, miny, maxx, maxy]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    total_area = width * height

    # Area per grid cell
    cell_area = total_area / n_samples

    # Side length of square grid cell rounded to nearest 10m
    cell_length = np.sqrt(cell_area)
    cell_length = np.ceil(cell_length / 10) * 10

    return cell_length


def get_iterative_grts_sample(points_gdf, n_samples, grid_output_path, decrement_factor=0.5, max_iterations=5):
    """
    Perform GRTS sampling with multiple attempts at different grid sizes.
    
    Parameters:
    -----------
    points_gdf : GeoDataFrame
        Points to sample from
    n_samples : int
        Number of samples to select
    grid_output_path : str
        Path to save the grid
    decrement_factor : float
        Factor to reduce grid size by in each iteration
    max_iterations : int
        Maximum number of iterations to try
        
    Returns:
    --------
    GeoDataFrame
        Sampled points
    """
    if n_samples <= 0:
        raise ValueError(f"Number of samples must be > 0, got {n_samples}")
    
    if len(points_gdf) == 0:
        raise ValueError("Cannot sample from empty GeoDataFrame")
    
    if n_samples > len(points_gdf):
        print(f"Warning: Requested {n_samples} samples but only {len(points_gdf)} points available")
        # Fall back to taking all points if we're asking for more than available
        return points_gdf.copy()

    # Compute initial grid max_length based on df bounding box and n_samples
    initial_length = compute_initial_length(points_gdf, n_samples)
    print(f"Initial grid max_length: {initial_length}")

    print(f"Sampling {n_samples} points from {len(points_gdf)} points")

    # Try different grid sizing approaches
    grid_sizes = []
    
    # Add the computed initial length and its decrements
    current_length = initial_length
    for _ in range(max_iterations):
        grid_sizes.append(current_length)
        current_length *= decrement_factor
    
    # Sort grid sizes from largest to smallest
    grid_sizes = sorted(grid_sizes, reverse=True)
    
    # Try each grid size
    for i, grid_length in enumerate(grid_sizes):
        try:
            print(f"Attempt {i+1}/{len(grid_sizes)} - Grid max_length: {grid_length}")
            
            # Create a QuadTree from the points
            qt = pygrts.QuadTree(points_gdf)
            
            # Split the grid
            qt.split_recursive(max_length=grid_length)
            
            # Perform GRTS sampling
            try:
                sampled_gdf = qt.sample(n_samples, samples_per_grid=1, random_state=RANDOM_SEED)
                
                # Check if we got the right number of samples
                if len(sampled_gdf) == n_samples:
                    # Save the grid with overwrite=True to handle existing files
                    save_grts_grid(qt, grid_output_path, overwrite=True)
                    print(f"Successfully sampled {n_samples} points with grid size {grid_length}")
                    return sampled_gdf, grid_length
                else:
                    print(f"  Got {len(sampled_gdf)} samples, needed {n_samples}")
            except Exception as e:
                print(f"  Error during sampling: {str(e)}")
                continue
                
        except Exception as e:
            print(f"  Error creating QuadTree: {str(e)}")
            continue
    
    print("All GRTS attempts failed, falling back to random sampling")
    return 