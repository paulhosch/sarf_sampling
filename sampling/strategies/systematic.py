import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import rasterio
from config import LABEL_BAND_NAME, PROJECTION_METERS, RANDOM_SEED
from sampling.io.input_image import get_input_image
from sampling.io.points import save_points_df
from sampling.utils.stats import update_sampling_stats

# Function to create systematic grid using meshgrid for a stratum
def sample_stratum_systematically(mask, n_samples_desired, max_iterations=5, decrement_factor=0.7, 
                                 training_samples_dirs=None, check_duplicates=False, transform=None, crs=None):
    if np.sum(mask) == 0:
        print(f"Warning: No pixels found for stratum")
        return np.array([]), np.array([]), None
    
    # Get indices of pixels in this stratum
    rows, cols = np.where(mask)
    
    if len(rows) <= n_samples_desired:
        print(f"Warning: Only {len(rows)} pixels available, returning all")
        return rows, cols, None
    
    # Get bounding box of pixels
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Calculate initial grid size based on desired number of samples
    area = (max_row - min_row + 1) * (max_col - min_col + 1)
    spacing = np.sqrt(area / n_samples_desired)
    
    # Round to nearest integer
    spacing = max(1, int(np.round(spacing)))
    
    print(f"Initial grid spacing: {spacing}")
    
    # Try different spacings in an iterative process
    best_sampled_rows = np.array([])
    best_sampled_cols = np.array([])
    best_count = 0
    best_spacing = None
    
    # Load existing training samples if requested
    existing_points = None
    if check_duplicates and training_samples_dirs is not None:
        print(f"Checking for duplicates in training samples directories")
        existing_points = load_existing_training_samples(training_samples_dirs)
        if existing_points is not None:
            print(f"Loaded {len(existing_points)} existing training points to check for duplicates")
    
    for iteration in range(max_iterations):
        print(f"Attempt {iteration+1}/{max_iterations} - Grid spacing: {spacing}")
        
        # Create meshgrid
        grid_rows = np.arange(min_row, max_row + 1, spacing)
        grid_cols = np.arange(min_col, max_col + 1, spacing)
        grid_r, grid_c = np.meshgrid(grid_rows, grid_cols)
        
        # Flatten grid
        grid_rows_flat = grid_r.flatten()
        grid_cols_flat = grid_c.flatten()
        
        # Keep only grid points that fall on mask
        valid_grid_points = []
        for i in range(len(grid_rows_flat)):
            r, c = grid_rows_flat[i], grid_cols_flat[i]
            # Check if indices are within image bounds
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                if mask[r, c]:
                    valid_grid_points.append((r, c))
        
        print(f"Found {len(valid_grid_points)} potential grid points")
        
        # Filter duplicate points if requested
        if check_duplicates and existing_points is not None and valid_grid_points:
            filtered_grid_points = remove_duplicate_points(valid_grid_points, existing_points, 
                                                          mask.shape, transform, crs)
            print(f"Removed {len(valid_grid_points) - len(filtered_grid_points)} duplicate points")
            valid_grid_points = filtered_grid_points
        
        # Convert to arrays
        if valid_grid_points:
            sampled_rows, sampled_cols = zip(*valid_grid_points)
            sampled_rows = np.array(sampled_rows)
            sampled_cols = np.array(sampled_cols)
        else:
            sampled_rows = np.array([])
            sampled_cols = np.array([])
        
        print(f"  Got {len(sampled_rows)} samples, needed {n_samples_desired}")
        
        # If we got exactly the number we want, return these samples
        if len(sampled_rows) == n_samples_desired:
            print(f"Successfully sampled {n_samples_desired} points with spacing {spacing}")
            return sampled_rows, sampled_cols, spacing
        
        # If we got more than we need, randomly select the desired number
        elif len(sampled_rows) > n_samples_desired:
            # Keep track of the best result but continue searching for exact matches
            if iteration == max_iterations - 1:
                print(f"Final iteration - randomly selecting {n_samples_desired} from {len(sampled_rows)} points")
                idx = np.random.RandomState(RANDOM_SEED).choice(
                    len(sampled_rows), n_samples_desired, replace=False
                )
                return sampled_rows[idx], sampled_cols[idx], spacing
            
            # Track this as our best result so far
            best_sampled_rows = sampled_rows
            best_sampled_cols = sampled_cols
            best_count = len(sampled_rows)
            best_spacing = spacing
            
            # Increase spacing for next iteration
            spacing = int(spacing / decrement_factor)
            print(f"  Too many samples, increasing spacing to {spacing}")
        
        # If we got fewer than we need, decrease spacing
        else:
            # Track as best result if it's better than what we had
            if len(sampled_rows) > best_count:
                best_sampled_rows = sampled_rows
                best_sampled_cols = sampled_cols
                best_count = len(sampled_rows)
                best_spacing = spacing
            
            # Decrease spacing for next iteration
            spacing = max(1, int(spacing * decrement_factor))
            print(f"  Too few samples, decreasing spacing to {spacing}")
    
    # After all iterations, return the best result we found
    print(f"Could not get exactly {n_samples_desired} samples. Best result: {best_count} samples")
    
    # If we have more than needed, randomly select
    if best_count > n_samples_desired:
        idx = np.random.RandomState(RANDOM_SEED).choice(
            best_count, n_samples_desired, replace=False
        )
        return best_sampled_rows[idx], best_sampled_cols[idx], best_spacing
    
    # Otherwise return all we found
    return best_sampled_rows, best_sampled_cols, best_spacing

def load_existing_training_samples(training_dirs):
    """
    Load existing training samples from geojson files in multiple training directories
    
    Parameters
    ----------
    training_dirs : str, Path, list of str, or list of Path
        Directory or list of directories containing training sample geojson files
        
    Returns
    -------
    gpd.GeoDataFrame or None
        Combined GeoDataFrame of all training samples, or None if no files found
    """
    # Ensure training_dirs is a list
    if not isinstance(training_dirs, list):
        training_dirs = [training_dirs]
    
    all_gdfs = []
    total_files = 0
    
    # Process each directory
    for training_dir in training_dirs:
        training_dir = Path(training_dir)
        if not training_dir.exists():
            print(f"Warning: Training directory {training_dir} does not exist")
            continue
        
        # Find all geojson files in this directory
        geojson_files = list(training_dir.glob("*.geojson"))
        if not geojson_files:
            print(f"No geojson files found in {training_dir}")
            continue
        
        print(f"Found {len(geojson_files)} geojson files in {training_dir}")
        total_files += len(geojson_files)
        
        # Load all files from this directory
        for file in geojson_files:
            try:
                gdf = gpd.read_file(file)
                all_gdfs.append(gdf)
                print(f"  Loaded {len(gdf)} points from {file.name}")
            except Exception as e:
                print(f"  Error loading {file}: {e}")
    
    # If no valid files were found, return None
    if not all_gdfs:
        print(f"No valid GeoJSON files found in any of the provided directories")
        return None
    
    # Combine all GeoDataFrames
    print(f"Combining {len(all_gdfs)} GeoDataFrames from {total_files} files")
    combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
    return combined_gdf

def remove_duplicate_points(valid_grid_points, existing_points, raster_shape, transform=None, crs=None):
    """
    Remove points from valid_grid_points that are already present in existing_points
    
    Parameters
    ----------
    valid_grid_points : list of tuples
        List of (row, col) tuples representing candidate grid points
    existing_points : gpd.GeoDataFrame
        GeoDataFrame containing existing training points
    raster_shape : tuple
        Shape of the raster (rows, cols)
    transform : affine.Affine, optional
        Affine transform of the raster, used to convert pixel to geographic coordinates
    crs : str or CRS, optional
        Coordinate reference system of the raster
    
    Returns
    -------
    list
        Filtered list of (row, col) tuples
    """
    if len(valid_grid_points) == 0 or existing_points is None or len(existing_points) == 0:
        return valid_grid_points
    
    # If we don't have transform and CRS, we can't do proper spatial comparison
    if transform is None or crs is None:
        print("Warning: Transform or CRS is missing - cannot check for spatial duplicates")
        return valid_grid_points
    
    # Convert pixel coordinates to geographical coordinates
    rows, cols = zip(*valid_grid_points)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    
    # Create GeoDataFrame for candidate points
    geometries = [Point(x, y) for x, y in zip(xs, ys)]
    candidates_gdf = gpd.GeoDataFrame(
        {'row': rows, 'col': cols},
        geometry=geometries,
        crs=crs
    )
    
    # Make sure both GeoDataFrames are in the same CRS
    if existing_points.crs != candidates_gdf.crs:
        existing_points = existing_points.to_crs(candidates_gdf.crs)
    
    # Use spatial index for efficient duplicate detection
    # We need to ensure we're working in a metric CRS for buffering
    # EPSG:4326 (WGS84) uses degrees, which makes buffer distances inconsistent
    metric_crs = PROJECTION_METERS  # From config.py, typically EPSG:3857
    
    # Convert both GeoDataFrames to a metric CRS for buffering
    candidates_metric = candidates_gdf.to_crs(metric_crs)
    existing_metric = existing_points.to_crs(metric_crs)
    
    # Now we can use a consistent buffer distance in meters
    buffer_distance = 1.0  # 1 meter buffer
    
    # Create a small buffer around existing points
    existing_buffered = existing_metric.copy()
    existing_buffered.geometry = existing_metric.geometry.buffer(buffer_distance)
    
    # Identify points that intersect with any buffered existing point
    spatial_index = existing_buffered.sindex
    
    # Get indices of candidate points that are near existing points
    duplicate_indices = set()
    for idx, candidate_geom in enumerate(candidates_metric.geometry):
        possible_matches_idx = list(spatial_index.intersection(candidate_geom.bounds))
        if possible_matches_idx:
            possible_matches = existing_buffered.iloc[possible_matches_idx]
            # More precise check using actual geometries
            if any(candidate_geom.intersects(match_geom) for match_geom in possible_matches.geometry):
                duplicate_indices.add(idx)
    
    # Remove duplicates
    non_duplicate_indices = [i for i in range(len(valid_grid_points)) if i not in duplicate_indices]
    filtered_points = [valid_grid_points[i] for i in non_duplicate_indices]
    
    print(f"Identified {len(duplicate_indices)} points that are duplicates of existing training points")
    
    return filtered_points

def systematic_sampling(raster_path, n_samples, strata_distribution, output_dir, grid_dir=None, 
                        max_iterations=5, decrement_factor=0.7, training_samples_dirs=None, 
                        check_duplicates=False):
    """
    Perform systematic sampling directly from a raster file using numpy meshgrid.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to the input raster file
    n_samples : int
        Total number of samples to select
    strata_distribution : str
        How to distribute samples between strata. Must be one of:
        - 'proportional': Samples distributed proportionally to stratum size
        - 'balanced': Equal number of samples per stratum
        - 'simple': No stratification, sample from all valid pixels
    output_dir : str or Path
        Path to save the sampled points
    grid_dir : str or Path, optional
        Path to save the grid visualization
    max_iterations : int, default=5
        Maximum number of iterations for adjusting grid spacing
    decrement_factor : float, default=0.7
        Factor by which to decrease grid spacing when too few points are found
    training_samples_dirs : str, Path, or list, optional
        Path to directory or list of directories containing existing training samples to check for duplicates
    check_duplicates : bool, default=False
        Whether to check for and remove duplicate points already in training samples
        
    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the sampled points with strata labels
    """
    # Convert paths to Path objects
    raster_path = Path(raster_path)
    output_dir = Path(output_dir)
    if grid_dir:
        grid_dir = Path(grid_dir)
        os.makedirs(grid_dir, exist_ok=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Print info about duplicate checking
    if check_duplicates and training_samples_dirs:
        if isinstance(training_samples_dirs, list):
            print(f"Will check for duplicates in {len(training_samples_dirs)} training sample directories")
        else:
            print(f"Will check for duplicates in training samples directory: {training_samples_dirs}")
    else:
        print("Duplicate checking is disabled")
    
    # Check if raster file exists
    if not raster_path.exists():
        raise FileNotFoundError(f"Input raster file not found: {raster_path}")
    
    # Read raster
    print(f"Loading raster from {raster_path}")
    image, metadata, label_band, osm_water_band = get_input_image(raster_path)
    
    # Get raster transform and CRS for spatial operations
    transform = metadata['transform']
    crs = metadata['crs']
    
    # Extract valid pixels (not NaN and not water)
    valid_mask = ~np.isnan(label_band) & (osm_water_band == 0)
    label_band_masked = np.where(valid_mask, label_band, np.nan)
    
    # Check if strata_distribution is valid
    if strata_distribution not in ['proportional', 'balanced', 'simple']:
        raise ValueError("Invalid strata distribution, must be 'proportional', 'balanced', or 'simple'")
    
    # If simple distribution, sample from all valid pixels without stratification
    if strata_distribution == 'simple':
        print(f"Using 'simple' distribution - sampling from all valid pixels without stratification")
        # Count total valid pixels
        total_valid_pixels = np.sum(valid_mask)
        print(f"Total valid pixels: {total_valid_pixels}")
        
        # Sample directly from all valid pixels
        rows, cols, spacing = sample_stratum_systematically(
            valid_mask, n_samples, max_iterations=max_iterations, 
            decrement_factor=decrement_factor,
            training_samples_dirs=training_samples_dirs, 
            check_duplicates=check_duplicates,
            transform=transform,
            crs=crs
        )
        
        # Get labels for the sampled pixels
        sampled_labels = label_band[rows, cols]
        
        # Convert pixel coordinates to geographic coordinates
        transform = metadata['transform']
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        
        # Create GeoDataFrame
        geometry = [Point(x, y) for x, y in zip(xs, ys)]
        combined_sample = gpd.GeoDataFrame(
            {LABEL_BAND_NAME: sampled_labels},
            geometry=geometry,
            crs=metadata['crs']
        )
        
        # Add sample_id column
        combined_sample['sample_id'] = range(len(combined_sample))
        
        # Convert to projection if needed
        combined_sample = combined_sample.to_crs(PROJECTION_METERS)
        
        # Print number of samples per stratum
        print(f"Number of total samples: {len(combined_sample)}")
        print(f"Number of samples per stratum: {combined_sample[LABEL_BAND_NAME].value_counts().to_dict()}")
        
        # Save the combined sample
        output_filename = f"{strata_distribution}_systematic.geojson"
        output_path = output_dir / output_filename
        save_points_df(combined_sample, output_path, overwrite=True)
        print(f"Saved combined samples to {output_path}")
        
        # Update sampling statistics
        update_sampling_stats(
            output_dir=output_dir,
            sampling_method="systematic",
            strata_distribution=strata_distribution,
            sampled_gdf=combined_sample,
            output_filename=output_filename,
            label_column=LABEL_BAND_NAME,
            simple_grid_length=spacing
        )
        
        return combined_sample
    
    # For proportional and balanced distribution, continue with stratification
    # Separate pixels by strata
    stratum_0_mask = (label_band_masked == 0)
    stratum_1_mask = (label_band_masked == 1)
    
    # Count pixels per stratum
    n_stratum_0 = np.sum(stratum_0_mask)
    n_stratum_1 = np.sum(stratum_1_mask)
    total_valid_pixels = n_stratum_0 + n_stratum_1
    
    print(f"Number of non-flooded pixels (stratum 0): {n_stratum_0}")
    print(f"Number of flooded pixels (stratum 1): {n_stratum_1}")
    
    # Calculate number of samples per stratum
    if strata_distribution == 'proportional':
        n_samples_0 = int(n_samples * (n_stratum_0 / total_valid_pixels))
        n_samples_1 = n_samples - n_samples_0
        print(f"Number of proportional non-flooded samples: {n_samples_0}")
        print(f"Number of proportional flooded samples: {n_samples_1}")
    else:  # balanced
        n_samples_0 = n_samples // 2
        n_samples_1 = n_samples - n_samples_0
        print(f"Number of balanced non-flooded samples: {n_samples_0}")
        print(f"Number of balanced flooded samples: {n_samples_1}")
    
    # Sample each stratum
    rows_0, cols_0, spacing_0 = sample_stratum_systematically(
        stratum_0_mask, n_samples_0, max_iterations=max_iterations, 
        decrement_factor=decrement_factor,
        training_samples_dirs=training_samples_dirs, 
        check_duplicates=check_duplicates,
        transform=transform,
        crs=crs
    )
    
    rows_1, cols_1, spacing_1 = sample_stratum_systematically(
        stratum_1_mask, n_samples_1, max_iterations=max_iterations, 
        decrement_factor=decrement_factor,
        training_samples_dirs=training_samples_dirs, 
        check_duplicates=check_duplicates,
        transform=transform,
        crs=crs
    )
    

    # TODO: remove this logic and just return an error if one stratum has too few samples
    # If one stratum has too few samples, try to add more from the other stratum
    if len(rows_0) < n_samples_0 and len(rows_1) > 0:
        deficit = n_samples_0 - len(rows_0)
        print(f"Deficit of {deficit} samples in stratum 0, trying to compensate from stratum 1")
        n_samples_1 += deficit
        rows_1, cols_1, spacing_1 = sample_stratum_systematically(
            stratum_1_mask, n_samples_1, max_iterations=max_iterations, 
            decrement_factor=decrement_factor,
            training_samples_dirs=training_samples_dirs, 
            check_duplicates=check_duplicates,
            transform=transform,
            crs=crs
        )
    
    if len(rows_1) < n_samples_1 and len(rows_0) > 0:
        deficit = n_samples_1 - len(rows_1)
        print(f"Deficit of {deficit} samples in stratum 1, trying to compensate from stratum 0")
        n_samples_0 += deficit
        rows_0, cols_0, spacing_0 = sample_stratum_systematically(
            stratum_0_mask, n_samples_0, max_iterations=max_iterations, 
            decrement_factor=decrement_factor,
            training_samples_dirs=training_samples_dirs, 
            check_duplicates=check_duplicates,
            transform=transform,
            crs=crs
        )
    
    # Combine results
    all_rows = np.concatenate([rows_0, rows_1]) if len(rows_0) > 0 and len(rows_1) > 0 else (rows_0 if len(rows_0) > 0 else rows_1)
    all_cols = np.concatenate([cols_0, cols_1]) if len(cols_0) > 0 and len(cols_1) > 0 else (cols_0 if len(cols_0) > 0 else cols_1)
    all_labels = np.concatenate([np.zeros(len(rows_0), dtype=int), np.ones(len(rows_1), dtype=int)]) if len(rows_0) > 0 and len(rows_1) > 0 else (np.zeros(len(rows_0), dtype=int) if len(rows_0) > 0 else np.ones(len(rows_1), dtype=int))
    
    # Check if we have any samples
    if len(all_rows) == 0:
        raise ValueError("Could not sample any points from either stratum")
    
    # Convert pixel coordinates to geographic coordinates
    transform = metadata['transform']
    xs, ys = rasterio.transform.xy(transform, all_rows, all_cols, offset='center')
    
    # Create GeoDataFrame
    geometry = [Point(x, y) for x, y in zip(xs, ys)]
    combined_sample = gpd.GeoDataFrame(
        {LABEL_BAND_NAME: all_labels},
        geometry=geometry,
        crs=metadata['crs']
    )
    
    # Add sample_id column
    combined_sample['sample_id'] = range(len(combined_sample))
    
    # Convert to projection if needed
    combined_sample = combined_sample.to_crs(PROJECTION_METERS)
    
    # Print number of samples per stratum
    print(f"Number of total samples: {len(combined_sample)}")
    print(f"Number of samples per stratum: {combined_sample[LABEL_BAND_NAME].value_counts().to_dict()}")
    
    # Save the combined sample
    output_filename = f"{strata_distribution}_systematic.geojson"
    output_path = output_dir / output_filename
    save_points_df(combined_sample, output_path, overwrite=True)
    print(f"Saved combined samples to {output_path}")
    
    # Update sampling statistics
    update_sampling_stats(
        output_dir=output_dir,
        sampling_method="systematic",
        strata_distribution=strata_distribution,
        sampled_gdf=combined_sample,
        output_filename=output_filename,
        label_column=LABEL_BAND_NAME,
        flooded_grid_length=spacing_1,
        non_flooded_grid_length=spacing_0
    )
    
    return combined_sample
