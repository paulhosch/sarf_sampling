import os
from pathlib import Path
from typing import Union

import pygrts

from config import PROJECTION_DEGREES

def save_grts_grid(qt: pygrts.QuadTree, output_path: Union[str, Path], overwrite: bool = True) -> None:
    """
    Save the GRTS grid to a GeoJSON file.
    
    Parameters:
    -----------
    qt : pygrts.QuadTree
        The GRTS grid to save
    output_path : str or Path
        Path to save the grid
    overwrite : bool, default=True
        Whether to overwrite the file if it already exists
    """
    # Convert the grid to a GeoDataFrame
    grid_df = qt.to_frame()
    
    # Ensure output directory exists
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Check if file exists and handle accordingly
    if output_path.exists() and not overwrite:
        print(f"File {output_path} already exists and overwrite=False. Skipping.")
        return
    
    # Convert to EPSG:4326 and save to GeoJSON
    grid_df.to_crs(PROJECTION_DEGREES).to_file(output_path, driver='GeoJSON')
    print(f"Saved GRTS grid to {output_path} in EPSG:4326")

def save_systematic_grid(grid_gdf, output_path, overwrite=True):
    """
    Save the systematic grid points to a file.
    
    Parameters:
    -----------
    grid_gdf : GeoDataFrame
        The grid points
    output_path : str
        Path to save the grid
    overwrite : bool
        Whether to overwrite existing file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if file exists
    if os.path.exists(output_path) and not overwrite:
        print(f"File {output_path} already exists. Set overwrite=True to overwrite.")
        return
    
    # Save grid
    grid_gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved systematic grid to {output_path}")