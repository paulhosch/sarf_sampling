"""
Utilities for sampling raster bands at point locations.
"""

import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen
from pathlib import Path
from shapely.geometry import Point
from config import LABEL_BAND_NAME

def sample_at_locations(
    input_image_path, 
    sample_locations_path=None,
    locations_dir=None, 
    output_dir=None,
    band_selection=None
):
    """
    Sample raster band values at point locations and create CSV files for ML training.
    
    Parameters
    ----------
    input_image_path : str or Path
        Path to the input raster image
    sample_locations_path : str, Path, or list, optional
        Path to a specific GeoJSON file with sample locations,
        or list of paths to multiple GeoJSON files
    locations_dir : str or Path, optional
        Directory containing GeoJSON files with sample locations
        (used if sample_locations_path is None)
    output_dir : str or Path, optional
        Directory to save output CSV files (created if it doesn't exist)
    band_selection : list, optional
        List of band names to sample. If None, all bands are sampled.

    
    Returns
    -------
    list
        List of paths to the created CSV files
    
    Notes
    -----
    This function samples raster band values at point locations defined in GeoJSON files.
    It verifies that the LABEL in the raster matches the LABEL in the GeoJSON.
    The output CSV files contain all band values and the label, ready for ML training.
    """
    # Ensure paths are Path objects
    input_image_path = Path(input_image_path)
    
    if output_dir is None:
        output_dir = Path.cwd() / "samples"
    else:
        output_dir = Path(output_dir) / "samples"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of GeoJSON files to process
    geojson_files = []
    
    if sample_locations_path is not None:
        if isinstance(sample_locations_path, (list, tuple)):
            geojson_files = [Path(p) for p in sample_locations_path]
        else:
            geojson_files = [Path(sample_locations_path)]
    elif locations_dir is not None:
        locations_dir = Path(locations_dir)
        geojson_files = list(locations_dir.glob("*.geojson"))
    else:
        raise ValueError("Either sample_locations_path or locations_dir must be provided")
    
    if not geojson_files:
        raise ValueError(f"No GeoJSON files found at the specified location(s)")
    
    print(f"Found {len(geojson_files)} GeoJSON file(s) to process")
    
    # Open the raster image
    with rasterio.open(input_image_path) as src:
        # Get band descriptions/names
        band_names = [
            src.descriptions[i-1] if i-1 < len(src.descriptions) and src.descriptions[i-1] else f"Band_{i}"
            for i in range(1, src.count + 1)
        ]
        
        # Find the label band index
        label_band_idx = None
        for i, name in enumerate(band_names):
            if LABEL_BAND_NAME in name:
                label_band_idx = i + 1  # Convert to 1-indexed
                break
        
        if label_band_idx is None:
            print(f"Warning: Label band '{LABEL_BAND_NAME}' not found in raster. Verification will be skipped.")
        
        # Filter to selected bands if specified
        if band_selection is not None:
            selected_bands = []
            for i, name in enumerate(band_names):
                if any(selected in name for selected in band_selection):
                    selected_bands.append(i + 1)  # Convert to 1-indexed
            
            if not selected_bands:
                print("Warning: None of the selected bands were found. Using all bands.")
                selected_bands = list(range(1, src.count + 1))
                selected_band_names = band_names
            else:
                selected_band_names = [band_names[i-1] for i in selected_bands]
        else:
            selected_bands = list(range(1, src.count + 1))
            selected_band_names = band_names
        
        print(f"Selected bands: {selected_band_names}")
        
        # Process each GeoJSON file
        output_files = []
        
        for geojson_file in geojson_files:
            print(f"Processing {geojson_file.name}...")
            
            # Load the GeoJSON file
            gdf = gpd.read_file(geojson_file)
            
            # Ensure GDF is in the same CRS as the raster
            if gdf.crs != src.crs:
                print(f"Reprojecting points from {gdf.crs} to {src.crs}")
                gdf = gdf.to_crs(src.crs)
            
            # Extract coordinates
            coords = [(geom.x, geom.y) for geom in gdf.geometry]
            
            # Sample raster values at coordinates
            sample_values = list(sample_gen(src, coords, indexes=selected_bands))
            
            # Create dataframe with sampled values
            data = []
            
            for i, (point, values) in enumerate(zip(gdf.itertuples(), sample_values)):
                # Get values as a regular Python list
                values_list = values.tolist()
                
                # Create row with band values and label
                row = {name: val for name, val in zip(selected_band_names, values_list)}
                
                # Add index and label from GeoJSON
                row['sample_id'] = getattr(point, 'sample_id', i)
                
                # Add label from GeoJSON if it exists
                if 'label' in gdf.columns:
                    row['label_from_points'] = getattr(point, 'label', None)
                
                # Add coordinates
                row['x'] = coords[i][0]
                row['y'] = coords[i][1]
                
                data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data)
            
            # Verify label if possible
            if label_band_idx is not None and 'label_from_points' in df.columns:
                # Find the label band in the selected bands
                label_col = None
                for col in df.columns:
                    if LABEL_BAND_NAME in col:
                        label_col = col
                        break
                
                if label_col is not None:
                    # Compare labels
                    matches = (df[label_col] == df['label_from_points']).sum()
                    total = len(df)
                    match_percentage = (matches / total) * 100
                    
                    print(f"Label verification: {matches}/{total} points match ({match_percentage:.2f}%)")
                    
                    # Rename the column for clarity
                    df.rename(columns={label_col: 'label_from_raster'}, inplace=True)
                    
                    # Use label from points as the final label
                    df['label'] = df['label_from_points']
                else:
                    print("Warning: Label band not found in selected bands. Verification skipped.")
            
            # Save as CSV
            output_filename = f"{geojson_file.stem}_samples.csv"
            output_path = output_dir / output_filename
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} samples to {output_path}")
            
            output_files.append(output_path)
        
        print(f"Finished processing {len(geojson_files)} files.")
        return output_files


def verify_sampled_points(csv_path, input_image_path):
    """
    Verify that sampled points in a CSV file match the raster values.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file with sampled points
    input_image_path : str or Path
        Path to the input raster image
    
    Returns
    -------
    dict
        Dictionary with verification results
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Check if coordinates exist
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV file must contain 'x' and 'y' columns")
    
    # Create points from coordinates
    geometry = [Point(xy) for xy in zip(df['x'], df['y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    
    # Open raster
    with rasterio.open(input_image_path) as src:
        # Get band descriptions/names
        band_names = [
            src.descriptions[i-1] if i-1 < len(src.descriptions) and src.descriptions[i-1] else f"Band_{i}"
            for i in range(1, src.count + 1)
        ]
        
        # Find the label band index
        label_band_idx = None
        for i, name in enumerate(band_names):
            if LABEL_BAND_NAME in name:
                label_band_idx = i + 1  # Convert to 1-indexed
                break
        
        if label_band_idx is None:
            return {"error": f"Label band '{LABEL_BAND_NAME}' not found in raster"}
        
        # Ensure GDF is in the same CRS as the raster
        if gdf.crs != src.crs and gdf.crs is not None:
            gdf = gdf.to_crs(src.crs)
        
        # Extract coordinates
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        
        # Sample raster values at coordinates
        sample_values = list(sample_gen(src, coords, indexes=[label_band_idx]))
        
        # Extract label values
        label_values = [val[0] for val in sample_values]
        
        # Compare with values in CSV
        if 'label' in df.columns:
            matches = sum(int(a) == int(b) for a, b in zip(label_values, df['label']))
            total = len(df)
            match_percentage = (matches / total) * 100
            
            return {
                "matches": matches,
                "total": total,
                "match_percentage": match_percentage,
                "csv_path": str(csv_path)
            }
        else:
            return {"error": "No 'label' column found in CSV file"}
