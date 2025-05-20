# %% Import
import os
from pathlib import Path
from sampling.strategies.wrapper import get_sample_locations
from sampling.processing.raster_to_points import raster_to_sampling_points
from sampling.vis.plot_multiple_samples import plot_multiple_samples
from sampling.processing.sample_at_locations import sample_at_locations

# %% User Input
sample_sizes = [100, 500, 1000]
site_id = "oder"
base_dir = "../../data"
input_image_path = Path(base_dir) / 'case_studies' / site_id / 'input_image'/ 'input_image.tif'

# %% Convert Image raster to points
points_path = Path(base_dir) / 'case_studies' / site_id / 'samples' / 'sampling_points.geojson'
#points_path = raster_to_sampling_points(input_image_path, points_path)
# %% Sample 

for n_samples in sample_sizes:
    output_dir = Path(base_dir) / 'case_studies' / site_id / 'samples'/ 'training' / str(n_samples)
    os.makedirs(output_dir, exist_ok=True)

    # 2. Get location where samples should be taken
    strata_distributions = ['simple', 'proportional', 'balanced']
    sampling_methods = ['random', 'grts', 'systematic']
    for strata_distribution in strata_distributions:
        for sampling_method in sampling_methods:
            get_sample_locations(
                image_path=input_image_path,
                points_path=points_path,
                n_samples=n_samples,
                strata_distribution=strata_distribution,
                sampling_method=sampling_method,
                output_dir=output_dir,
                decrement_factor=0.7,
                max_iterations=20
            )

    # 3. Sample at locations
    sample_at_locations(
        input_image_path=input_image_path,
        locations_dir=output_dir / 'locations',
        output_dir=output_dir
    )

# %% Plot all samples of one sample size in one figure

    sample_size_to_plot = 1000
    output_dir = Path(base_dir) / 'case_studies' / site_id / 'samples'/ 'training' / str(sample_size_to_plot)
    output_name = str(sample_size_to_plot) + '_sample_sets.png'
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
    fig = plot_multiple_samples(
        input_image_path=input_image_path,
        sample_paths=sample_paths,
        stats_path=output_dir / 'locations' / 'stats.csv',
        custom_extent='square',
        ncols=3,
        figsize=(15, 15),
        output_path=output_dir / output_name 
    ) 
# %%
