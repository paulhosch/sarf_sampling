import geopandas as gpd
from pathlib import Path
import os
from typing import Union

def save_points_df(gdf: gpd.GeoDataFrame, output_path: Union[str, Path], overwrite: bool = True) -> None:
    """
    Save a GeoDataFrame to GeoJSON format in EPSG:4326 coordinate system.
    
    Parameters:
    -----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to save
    output_path : str or Path
        Path to save the points
    overwrite : bool, default=True
        Whether to overwrite the file if it already exists
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Check if file exists and handle accordingly
    if output_path.exists() and not overwrite:
        print(f"File {output_path} already exists and overwrite=False. Skipping.")
        return
    
    # Convert to EPSG:4326 and save
    gdf.to_crs("EPSG:4326").to_file(output_path, driver='GeoJSON')
    print(f"Saved points to {output_path} in EPSG:4326")