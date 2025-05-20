# %% Import
import os
from pathlib import Path

from sampling.strategies.wrapper import get_sample_locations
from sampling.processing.raster_to_points import raster_to_sampling_points
from sampling.vis.plot_multiple_samples import plot_multiple_samples
from sampling.processing.sample_at_locations import sample_at_locations


# %% User Input

n_samples = 1000           
site_id = "oder"     

base_dir = "../../data" 
input_image_path = Path(base_dir) / 'case_studies' / site_id / 'input_image'/ 'input_image.tif'

output_dir = Path(base_dir) / 'case_studies' / site_id / 'sample_sets'/ 'training'/ str(n_samples)
os.makedirs(output_dir, exist_ok=True)
points_path = Path(base_dir) / 'case_studies' / site_id / 'samples' / 'sampling_points.geojson'

# %% 1. Convert Image raster to points 
# Vectorizing the whole raster is a workaround to make use of the pygrts pacakge and is not very efficient and unneccesary
# this should be improved in future versions

points_path = raster_to_sampling_points(input_image_path, points_path)

# %% 2. Get location where samples should be taken 
# This wrapper allows to sample using different class and spatial distributions
# the code behind this wrapper is not pretty and should be restructured in future versions

# the following class/strata distributions are available:
strata_distributions = ['simple', 'proportional', 'balanced']

# the following sampling methods are available:
sampling_methods = ['random', 'grts', 'systematic']

for strata_distribution in strata_distributions:
    for sampling_method in sampling_methods:
        systematic_samples = get_sample_locations(
            image_path=input_image_path,
            points_path=points_path,
            n_samples=n_samples,
            strata_distribution=strata_distribution,
            sampling_method=sampling_method,
            output_dir=output_dir,
            decrement_factor=0.7, # Controls how aggressively spacing changes between iterations:
                                  # - Smaller values (0.3-0.5): Faster convergence but may overshoot optimal spacing
                                  # - Larger values (0.6-0.9): Slower but more stable convergence 
                                  # - When too many samples: spacing increases (÷ by factor)
                                  # - When too few samples: spacing decreases (× by factor)
            max_iterations=20,    # for most accurate grid, set decrement_factor and max_iterations as high as computationally feasible
)

# a stats.csv file documents the number of samples per stratum 

# %% 3. Sample at locations
csv_files = sample_at_locations(
    input_image_path=input_image_path,
    locations_dir=output_dir / 'locations',
    output_dir=output_dir
)
# %% 3. Plot 
# Plot all samples in one figure (or only one sample at a time)
sample_paths = [
    output_dir / 'locations' / 'simple_random.geojson',
    output_dir / 'locations' / 'proportional_random.geojson',
    output_dir / 'locations' / 'balanced_random.geojson',
    output_dir / 'locations' / 'simple_grts.geojson',
    output_dir / 'locations' / 'proportional_grts.geojson',
    output_dir / 'locations' / 'balanced_grts.geojson',
    output_dir / 'locations' / 'simple_systematic.geojson',
    output_dir / 'locations' / 'proportional_systematic.geojson',
    output_dir / 'locations' / 'balanced_systematic.geojson',
   
]

# Corresponding grid paths (None, single path, or tuple of two paths for stratified grts)
grid_paths = [
    None,  # No grid for simple random
    None, # No grid for proportional random
    None, # No grid for balanced random
    output_dir /'locations' / 'grids' / 'simple_grts_grid.geojson', 
    (output_dir /'locations' / 'grids' / 'flooded_grts_grid.geojson', output_dir / 'locations' / 'grids' / 'non_flooded_grts_grid.geojson'),
    (output_dir / 'locations' / 'grids' / 'flooded_grts_grid.geojson', output_dir / 'locations' / 'grids' / 'non_flooded_grts_grid.geojson'), 
    None, # No grid for simple systematic
    None, # No grid for proportional systematic
    None # No grid for balanced systematic
]

# Figure with all samples compared in a grid
fig = plot_multiple_samples(
    input_image_path=input_image_path,
    sample_paths=sample_paths,
    stats_path=output_dir / 'locations' / 'stats.csv',
    custom_extent='square',#[-0.4, -0.3, 39.3, 39.4],
    #grid_paths=grid_paths, # optinional, comment out for no grid display 
    ncols=3,
    figsize=(15, 15),
    output_path=output_dir / 'sample_sets.svg' #reduce dpi  for png export 
);





# %%
