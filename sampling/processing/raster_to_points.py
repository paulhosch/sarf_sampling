from config import LABEL_BAND_NAME, WATER_BAND_NAME
import rasterio
import geopandas as gpd
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional, Union
from shapely.geometry import Point
from ..io.points import save_points_df
from ..io.input_image import get_input_image

def raster_to_sampling_points(input_image_path: str, output_path: str) -> gpd.GeoDataFrame:
    """
    Convert a raster to points GeoDataFrame for sampling
    
    Parameters:
    -----------
    input_image_path : str
        Path to the input image
    output_path : str, optional
        Path to save the GeoDataFrame. 
        
    Returns:
    --------
    gpd.GeoDataFrame
        GeoDataFrame with points and strata values
    """
    # Read the input image
    image, metadata, label_band, osm_water_band = get_input_image(input_image_path)

    # Valid pixels are those with LABEL data and not water
    valid_mask = ~np.isnan(label_band) & (osm_water_band == 0)
    rows, cols = np.where(valid_mask)
    
    # Convert pixel coordinates to geographic coordinates
    transform = metadata['transform']
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    
    # Create geometries
    geometry = [Point(x, y) for x, y in zip(xs, ys)]
    
    # Get strata values directly from label band
    strata_values = label_band[rows, cols].astype(int)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {LABEL_BAND_NAME: strata_values},
        geometry=geometry,
        crs=metadata['crs']
    )
    
    # Add a sample_id column 
    gdf['sample_id'] = range(len(gdf))
    
    print(f"Created GeoDataFrame with {len(gdf)} points")
    print(f"Points per stratum: {gdf[LABEL_BAND_NAME].value_counts().to_dict()}")
    
    # Save
    save_points_df(gdf, output_path)
    print(f"Saved points GeoDataFrame to {output_path}")

    return output_path 