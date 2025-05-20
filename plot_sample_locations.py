# %% Import
import os
from pathlib import Path
from sampling.vis.plot_multiple_samples import plot_multiple_samples

n_samples = 100           
site_id = "valencia"
iteration_id = "iteration_1"

base_dir = "../../data" 
samples_dir = Path(base_dir) / 'case_studies' / site_id / 'samples'/ iteration_id / 'training' / str(n_samples)
input_image_path = Path(base_dir) / 'case_studies' / site_id / 'input_image'/ 'input_image.tif'
plot_dir = Path(base_dir) / 'case_studies' / site_id / 'samples'/ 'plots'
os.makedirs(plot_dir, exist_ok=True)

sample_paths = [
    samples_dir / 'locations' / 'simple_random.geojson',
    samples_dir / 'locations' / 'proportional_random.geojson',
    samples_dir / 'locations' / 'balanced_random.geojson',
    samples_dir / 'locations' / 'simple_grts.geojson',
    samples_dir / 'locations' / 'proportional_grts.geojson',
    samples_dir / 'locations' / 'balanced_grts.geojson',
    samples_dir / 'locations' / 'simple_systematic.geojson',
    samples_dir / 'locations' / 'proportional_systematic.geojson',
    samples_dir / 'locations' / 'balanced_systematic.geojson',
]

grid_paths = [
    None,  # No grid for simple random
    None, # No grid for proportional random
    None, # No grid for balanced random
    samples_dir /'locations' / 'grids' / 'simple_grts_grid.geojson', 
    (samples_dir /'locations' / 'grids' / 'flooded_grts_grid.geojson', samples_dir / 'locations' / 'grids' / 'non_flooded_grts_grid.geojson'),
    (samples_dir / 'locations' / 'grids' / 'flooded_grts_grid.geojson', samples_dir / 'locations' / 'grids' / 'non_flooded_grts_grid.geojson'), 
    None, # No grid for simple systematic
    None, # No grid for proportional systematic
    None # No grid for balanced systematic
]


grid_paths = None # comment out to use grid_paths above
fig = plot_multiple_samples(
    input_image_path=input_image_path,
    sample_paths=sample_paths,
    stats_path=samples_dir / 'locations' / 'stats.csv',
    custom_extent='square',
    #grid_paths=grid_paths, # uncomment to show grids
    ncols=3,
    figsize=(15, 15),
    output_path=plot_dir / f'sample_sets_{iteration_id}_{site_id}_{n_samples}.png'
) 
# %%
