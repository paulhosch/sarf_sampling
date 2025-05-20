# %% Import
import os
from pathlib import Path

from sampling.strategies.wrapper import get_sample_locations
from sampling.processing.raster_to_points import raster_to_sampling_points
from sampling.vis.plot_multiple_samples import plot_multiple_samples
from sampling.processing.sample_at_locations import sample_at_locations

# %% User Input

n_samples = 1000           
site_ids = ["valencia", "danube", "oder"]
base_dir = "../../data" 

strata_distributions = ['proportional', 'balanced']
sampling_methods = ['systematic']



# %% 2. Get location where samples should be taken 
for site_id in site_ids:
    input_image_path = Path(base_dir) / 'case_studies' / site_id / 'input_image'/ 'input_image.tif'

    output_dir = Path(base_dir) / 'case_studies' / site_id / 'samples'/ 'testing'/ str(n_samples)
    os.makedirs(output_dir, exist_ok=True)
    points_path = Path(base_dir) / 'case_studies' / site_id / 'samples'/ 'sampling_points.geojson'

    # training sample dirs for checking for duplicates
    training_sample_sizes = [100]
    training_samples_dirs = []
    for training_sample_size in training_sample_sizes:
        training_samples_dir = Path(base_dir) / 'case_studies' / site_id / 'sample_sets'/ 'training'/ str(training_sample_size) /'locations'
        training_samples_dirs.append(training_samples_dir)
    '''
    for strata_distribution in strata_distributions:
        for sampling_method in sampling_methods:
            systematic_samples = get_sample_locations(
                image_path=input_image_path,
                points_path=points_path,
                n_samples=n_samples,
                strata_distribution=strata_distribution,
                sampling_method=sampling_method,
                output_dir=output_dir,
                decrement_factor=0.5, # Controls how aggressively spacing changes between iterations:
                                    # - Smaller values (0.3-0.5): Faster convergence but may overshoot optimal spacing
                                    # - Larger values (0.6-0.9): Slower but more stable convergence 
                                    # - When too many samples: spacing increases (÷ by factor)
                                    # - When too few samples: spacing decreases (× by factor)
                max_iterations=20,    # for most accurate grid, set decrement_factor and max_iterations as high as computationally feasible
                training_samples_dirs=training_samples_dirs,
                check_duplicates=True
    )

   ''' 
    # a stats.csv file documents the number of samples per stratum 

    csv_files = sample_at_locations(
        input_image_path=input_image_path,
        locations_dir=output_dir / 'locations',
        output_dir=output_dir
)
# %% 3. Plot 
# Plot all samples in one figure (or only one sample at a time)
site_id = "valencia"
n_samples = 1000

output_dir = Path(base_dir) / 'case_studies' / site_id / 'samples'/ 'testing'/ str(n_samples)
input_image_path = Path(base_dir) / 'case_studies' / site_id / 'input_image'/ 'input_image.tif'

sample_paths = [

    output_dir / 'locations' / 'proportional_systematic.geojson',
    output_dir / 'locations' / 'balanced_systematic.geojson',
   
]


# Figure with all samples compared in a grid
fig = plot_multiple_samples(
    input_image_path=input_image_path,
    sample_paths=sample_paths,
    stats_path=output_dir / 'locations' / 'stats.csv',
    custom_extent='square',#[-0.4, -0.3, 39.3, 39.4],
    ncols=3,
    figsize=(15, 15),
    output_path=output_dir / 'sample_sets.svg' #reduce dpi  for png export 
);





# %%
