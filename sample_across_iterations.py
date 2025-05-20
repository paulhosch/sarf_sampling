# %% Import
import os
from pathlib import Path
from typing import List, Dict, Union, Optional
import random
import numpy as np
import time
from config import RANDOM_SEED

from sampling.strategies.wrapper import get_sample_locations
from sampling.processing.raster_to_points import raster_to_sampling_points
from sampling.vis.plot_multiple_samples import plot_multiple_samples
from sampling.processing.sample_at_locations import sample_at_locations

# %% Wrapper function
def sample_across_iterations(
    base_dir: str,
    site_ids: List[str],
    n_iterations: int,
    training_sample_sizes: List[int],
    testing_sample_size: int,
    training_strata_distributions: List[str] = None,
    training_sampling_methods: List[str] = None,
    testing_strata_distributions: List[str] = None,
    testing_sampling_methods: List[str] = None,
    decrement_factor: float = 0.7,
    max_iterations: int = 20,
    check_duplicates_for_testing: bool = True
):
    """
    Create training and testing samples across multiple iterations.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for data (e.g., "../../data")
    site_ids : List[str]
        List of site IDs to process (e.g., ["valencia", "oder", "danube"])
    n_iterations : int
        Number of iterations to perform
    training_sample_sizes : List[int]
        List of sample sizes for training (e.g., [100, 500, 1000])
    testing_sample_size : int
        Sample size for testing
    training_strata_distributions : List[str], optional
        Strata distributions for training (default: ['simple', 'proportional', 'balanced'])
    training_sampling_methods : List[str], optional
        Sampling methods for training (default: ['random', 'grts', 'systematic'])
    testing_strata_distributions : List[str], optional
        Strata distributions for testing (default: ['proportional', 'balanced'])
    testing_sampling_methods : List[str], optional
        Sampling methods for testing (default: ['systematic'])
    decrement_factor : float, optional
        Controls how aggressively spacing changes between iterations
    max_iterations : int, optional
        Maximum iterations for iterative sampling methods
    check_duplicates_for_testing : bool, optional
        Whether to check for duplicate samples between training and testing
    """
    # Set default values if not specified
    if training_strata_distributions is None:
        training_strata_distributions = ['simple', 'proportional', 'balanced']
    if training_sampling_methods is None:
        training_sampling_methods = ['random', 'grts', 'systematic']
    if testing_strata_distributions is None:
        testing_strata_distributions = ['proportional', 'balanced']
    if testing_sampling_methods is None:
        testing_sampling_methods = ['systematic']
    
    # Loop through iterations
    for iteration in range(1, n_iterations + 1):
        iteration_id = f"iteration_{iteration}"
        # Set a unique random seed for this iteration
        seed = (RANDOM_SEED if RANDOM_SEED is not None else 0) + iteration
        random.seed(seed)
        np.random.seed(seed)
        print(f"\nProcessing {iteration_id} with random seed {seed}...")
        
        for site_id in site_ids:
            print(f"\nSite: {site_id}")
            input_image_path = Path(base_dir) / 'case_studies' / site_id / 'input_image' / 'input_image.tif'
            points_path = Path(base_dir) / 'case_studies' / site_id / 'samples' / 'sampling_points.geojson'
            
            # Create points file if it doesn't exist
            if not points_path.exists():
                print(f"Creating sampling points for {site_id}...")
                points_path = raster_to_sampling_points(input_image_path, points_path)
            
            # 1. Training samples
            print(f"Creating training samples for {site_id}...")
            for n_samples in training_sample_sizes:
                print(f"  Sample size: {n_samples}")
                output_dir = Path(base_dir) / 'case_studies' / site_id / 'samples' / iteration_id / 'training' / str(n_samples)
                os.makedirs(output_dir, exist_ok=True)
                
                for strata_distribution in training_strata_distributions:
                    for sampling_method in training_sampling_methods:
                        strategy_name = f"{strata_distribution}_{sampling_method}"
                        print(f"    Strategy: {strategy_name}")
                        # Time the sampling process
                        start_time = time.time()
                        get_sample_locations(
                            image_path=input_image_path,
                            points_path=points_path,
                            n_samples=n_samples,
                            strata_distribution=strata_distribution,
                            sampling_method=sampling_method,
                            output_dir=output_dir,
                            decrement_factor=decrement_factor,
                            max_iterations=max_iterations
                        )
                
                # Sample at locations
                sample_at_locations(
                    input_image_path=input_image_path,
                    locations_dir=output_dir / 'locations',
                    output_dir=output_dir,
                    sampling_time_seconds=time.time() - start_time
                )
            
            # 2. Testing samples
            print(f"Creating testing samples for {site_id}...")
            output_dir = Path(base_dir) / 'case_studies' / site_id / 'samples' / iteration_id / 'testing' / str(testing_sample_size)
            os.makedirs(output_dir, exist_ok=True)
            
            # Find all training sample dirs for checking duplicates
            training_samples_dirs = []
            if check_duplicates_for_testing:
                for n_samples in training_sample_sizes:
                    training_samples_dir = Path(base_dir) / 'case_studies' / site_id / 'samples' / iteration_id / 'training' / str(n_samples) / 'locations'
                    training_samples_dirs.append(training_samples_dir)
            
            for strata_distribution in testing_strata_distributions:
                for sampling_method in testing_sampling_methods:
                    strategy_name = f"{strata_distribution}_{sampling_method}"
                    print(f"    Strategy: {strategy_name}")
                    # Time the sampling process
                    start_time = time.time()
                    get_sample_locations(
                        image_path=input_image_path,
                        points_path=points_path,
                        n_samples=testing_sample_size,
                        strata_distribution=strata_distribution,
                        sampling_method=sampling_method,
                        output_dir=output_dir,
                        decrement_factor=decrement_factor,
                        max_iterations=max_iterations,
                        training_samples_dirs=training_samples_dirs if check_duplicates_for_testing else None,
                        check_duplicates=check_duplicates_for_testing
                    )
            
            # Sample at locations
            sample_at_locations(
                input_image_path=input_image_path,
                locations_dir=output_dir / 'locations',
                output_dir=output_dir,
                sampling_time_seconds=time.time() - start_time
            )
    
    print("\nSampling across iterations completed!")

# %% Example usage
if __name__ == "__main__":
    # Parameters
    base_dir = "../../data"
    site_ids = ["valencia", "oder", "danube"]
    n_iterations = 10
    training_sample_sizes = [100, 500, 1000]
    testing_sample_size = 1000
    
    # For simplicity, we can specify a subset of strategies
    training_strata_distributions = ['simple', 'proportional', 'balanced']
    training_sampling_methods = ['random', 'systematic', 'grts']  
    testing_strata_distributions = ['proportional', 'balanced']
    testing_sampling_methods = ['systematic']
    
    # Run sampling across iterations
    sample_across_iterations(
        base_dir=base_dir,
        site_ids=site_ids,
        n_iterations=n_iterations,
        training_sample_sizes=training_sample_sizes,
        testing_sample_size=testing_sample_size,
        training_strata_distributions=training_strata_distributions,
        training_sampling_methods=training_sampling_methods,
        testing_strata_distributions=testing_strata_distributions,
        testing_sampling_methods=testing_sampling_methods,
        decrement_factor=0.7,
        max_iterations=20,
        check_duplicates_for_testing=True
    )

# %% 