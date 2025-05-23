# saRFlood-2 Sampling

This repository contains workflow for the 2. part of the saRFlood pipeline - Sampling from the casestudy input_image.
Following Sampling Strategies can be used:

- Simple Random Sampling
- Stratified Sampling
- Generalized Random Tessellation Stratified (GRTS) Sampling
- Systematic Square Grid Sampling

To compare different class and spatial distributions in the sample selection, the following sample sets can be created and plotted.

## Sample Sets

![Sample Sets](sample_sets_filled.png)
Data Sources: (c) OpenStreetMap contributors and (c) European Union, CEMS Rapid Mapping

## Usage

pipeline.py outlines the complete sampling and visualization process.

## References

### Packages

- PyGRTS ([GitHub](https://github.com/jsta/pygrts)) - Copyright (c) 2020-- pygrts

### Publications

Stevens Jr, D. L., & Olsen, A. R. (2004). Spatially balanced sampling of natural resources. _Journal of the American Statistical Association_, 99(465), 262-278. https://doi.org/10.1198/016214504000000250

# Installation

1. Create and activate the conda environment:
   ```sh
   conda create -n sarf_sampling python=3.10 -y
   conda activate sarf_sampling
   ```
2. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```
3. (Optional) Use as Jupyter kernel:
   ```sh
   pip install ipykernel
   python -m ipykernel install --user --name sarf_sampling --display-name "Python (sarf_sampling)"
   ```

# File Structure

The pipeline expects and produces files in the following structure:

```
<data>
└── case_studies/
    └── <site_id>/
        ├── input_image/
        │   └── input_image.tif         # Input raster image
        └── samples/
            ├── sampling_points.geojson # All possible sampling points (vectorized raster)
            └── <n_samples>/            # One folder per sample size
                └── locations/
                    ├── simple_random_samples.geojson
                    ├── proportional_random_samples.geojson
                    ├── balanced_random_samples.geojson
                    ├── simple_grts_samples.geojson
                    ├── proportional_grts_samples.geojson
                    ├── balanced_grts_samples.geojson
                    ├── simple_systematic_samples.geojson
                    ├── proportional_systematic_samples.geojson
                    ├── balanced_systematic_samples.geojson
                    ├── stats.csv
                    └── grids/
                        ├── simple_grts_grid.geojson
                        ├── flooded_grts_grid.geojson
                        └── non_flooded_grts_grid.geojson
                └── sample_sets.png     # Output plot for this sample size
```

- `<site_id>`: Name of the case study (e.g., `valencia`, `danube`)
- `<n_samples>`: Number of samples (e.g., `100`, `500`, `1000`)
- All outputs are organized by sample size for clarity and reproducibility.
