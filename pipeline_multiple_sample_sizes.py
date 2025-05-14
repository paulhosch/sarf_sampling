# %% Import
import os
from pathlib import Path
from sampling.strategies.wrapper import get_sample_locations
from sampling.processing.raster_to_points import raster_to_sampling_points
from sampling.vis.plot_multiple_samples import plot_multiple_samples
from sampling.processing.sample_at_locations import sample_at_locations

# %% User Input
sample_sizes = [100, 500, 1000]
site_id = "valencia"
base_dir = "../../data"
input_image_path = Path(base_dir) / 'case_studies' / site_id / 'input_image'/ 'input_image.tif'

# %% Loop over sample sizes
# Set points_path outside the loop, as it is independent of n_samples
points_path = Path(base_dir) / 'case_studies' / site_id / 'samples' / 'sampling_points.geojson'
points_path = raster_to_sampling_points(input_image_path, points_path)

for n_samples in sample_sizes:
    output_dir = Path(base_dir) / 'case_studies' / site_id / 'samples'/ str(n_samples)
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
                decrement_factor=0.5,
                max_iterations=5
            )

    # 3. Sample at locations
    sample_at_locations(
        input_image_path=input_image_path,
        locations_dir=output_dir / 'locations',
        output_dir=output_dir
    )

    # 4. Plot all samples in one figure
    sample_paths = [
        output_dir / 'locations' / 'simple_random_samples.geojson',
        output_dir / 'locations' / 'proportional_random_samples.geojson',
        output_dir / 'locations' / 'balanced_random_samples.geojson',
        output_dir / 'locations' / 'simple_grts_samples.geojson',
        output_dir / 'locations' / 'proportional_grts_samples.geojson',
        output_dir / 'locations' / 'balanced_grts_samples.geojson',
        output_dir / 'locations' / 'simple_systematic_samples.geojson',
        output_dir / 'locations' / 'proportional_systematic_samples.geojson',
        output_dir / 'locations' / 'balanced_systematic_samples.geojson',
    ]
    fig = plot_multiple_samples(
        input_image_path=input_image_path,
        sample_paths=sample_paths,
        stats_path=output_dir / 'locations' / 'stats.csv',
        custom_extent='square',
        ncols=3,
        figsize=(15, 15),
        output_path=output_dir / 'sample_sets.png'
    ) 