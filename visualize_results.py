# %% Import
from pathlib import Path
from sampling.vis.plot_multiple_samples import plot_multiple_samples

# %% User Input
n_samples = 100           
site_id = "valencia"     
base_dir = "../../data" 
output_dir = Path(base_dir) / 'case_studies' / site_id / 'samples'/ str(n_samples)
input_image_path = Path(base_dir) / 'case_studies' / site_id / 'input_image'/ 'input_image.tif'

# %% Define sample and grid paths
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

# %% Plot all samples in one figure
grid_paths = None # comment out to use grid_paths above
fig = plot_multiple_samples(
    input_image_path=input_image_path,
    sample_paths=sample_paths,
    stats_path=output_dir / 'locations' / 'stats.csv',
    custom_extent='square',
    #grid_paths=grid_paths, # uncomment to show grids
    ncols=3,
    figsize=(15, 15),
    output_path=output_dir / 'sample_sets.png'
) 