"""
Wrapper function for all sampling methods with a unified interface.
"""

import os
from pathlib import Path
import geopandas as gpd

def get_sample_locations(
    points_path=None,
    image_path=None,
    n_samples=100,
    strata_distribution='simple',
    sampling_method='random',
    output_dir=None,
    grid_dir=None,
    max_iterations=5,
    decrement_factor=0.5,
    training_samples_dirs=None,
    check_duplicates=False,
    **kwargs
):
    """
    Unified wrapper for all sampling methods.
    
    Parameters
    ----------
    points_path : str or Path, optional
        Path to points file (required for point-based methods)
    image_path : str or Path, optional
        Path to raster image file (required for direct-from-raster methods)
    n_samples : int, default=100
        Number of samples to generate
    strata_distribution : str, default='simple'
        How to distribute samples between strata:
        - 'simple': No stratification, all pixels treated equally
        - 'proportional': Samples distributed proportionally to stratum size
        - 'balanced': Equal number of samples per stratum
    sampling_method : str, default='random'
        Spatial distribution method:
        - 'random': Simple random sampling
        - 'grts': Generalized Random Tessellation Stratified sampling
        - 'systematic': Systematic grid sampling
    output_dir : str or Path, default=None
        Directory to save output files (created if doesn't exist)
    grid_dir : str or Path, default=None
        Directory to save grid files for GRTS/systematic methods
    max_iterations : int, default=5
        Maximum iterations for iterative methods
    decrement_factor : float, default=0.7
        Factor for decreasing spacing in iterative methods
    training_samples_dirs : str, Path, list of str, or list of Path, optional
        Directory or list of directories containing existing training samples to check for duplicates
    check_duplicates : bool, default=False
        Whether to check for and remove duplicate points that exist in training samples
    **kwargs : dict
        Additional keyword arguments passed to the specific sampling method
    
    Returns
    -------
    geopandas.GeoDataFrame
        The sampled points
    
    Raises
    ------
    ValueError
        If invalid parameters are provided
    """
    # Import here to avoid circular imports
    from sampling.strategies.random import random_sampling
    from sampling.strategies.grts import grts_sampling
    from sampling.strategies.systematic import systematic_sampling
    
    # Validate parameters
    if strata_distribution not in ['simple', 'proportional', 'balanced']:
        raise ValueError("strata_distribution must be 'simple', 'proportional', or 'balanced'")
    
    if sampling_method not in ['random', 'grts', 'systematic']:
        raise ValueError("sampling_method must be 'random', 'grts', or 'systematic'")
    
    # Set up directories
    output_dir = Path(output_dir) / 'locations'
        
    os.makedirs(output_dir, exist_ok=True)
    
    if grid_dir is None and sampling_method in ['grts', 'systematic']:
        grid_dir = output_dir / 'grids'
        os.makedirs(grid_dir, exist_ok=True)
    
    # Select appropriate sampling method
    if sampling_method == 'random':
        if points_path is None:
            raise ValueError("points_path is required for random sampling")
        
        return random_sampling(
            path_to_points=points_path,
            n_samples=n_samples,
            strata_distribution=strata_distribution,
            output_dir=output_dir,
            **kwargs
        )
    
    elif sampling_method == 'grts':
        if points_path is None:
            raise ValueError("points_path is required for GRTS sampling")
        
        return grts_sampling(
            points_path=points_path,
            n_samples=n_samples,
            strata_distribution=strata_distribution,
            output_dir=output_dir,
            grid_dir=grid_dir,
            decrement_factor=decrement_factor,
            max_iterations=max_iterations,
            **kwargs
        )
    
    elif sampling_method == 'systematic':
        # For systematic sampling, we can work with either points or raster
        if image_path is not None:
            # Direct from raster
            return systematic_sampling(
                raster_path=image_path,
                n_samples=n_samples,
                strata_distribution=strata_distribution,
                output_dir=output_dir,
                grid_dir=grid_dir,
                max_iterations=max_iterations,
                decrement_factor=decrement_factor,
                training_samples_dirs=training_samples_dirs,
                check_duplicates=check_duplicates,
                **kwargs
            )
        elif points_path is not None:
            # From points
            raise NotImplementedError(
                "Systematic sampling from points is not directly implemented. "
                "Use grts_sampling with sampling_strategy='systematic' instead."
            )
        else:
            raise ValueError("Either points_path or image_path is required for systematic sampling") 