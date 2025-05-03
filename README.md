# saRFlood-2 Sampling

This repository contains functionalities for the 2. part of the saRFlood pipeline - Sampling from the input_image.
Following Sampling Strategies can be used:

- Simple Random Sampling
- Stratified Sampling
- Generalized Random Tessellation Stratified (GRTS) Sampling
- Systematic Square Grid Sampling

To compare different the different class and spatial distributions, the following sample sets can be created and plotted:
![Sample Sets](sample_sets_filled.png)


Data Source: (c) OpenStreetMap contributors and (c) European Union, Copernicus Emergency

## References

Used packages:

- PyGRTS: Copyright (c) 2020-- pygrts

@article{stevens_olsen_2004,
title={Spatially balanced sampling of natural resources},
author={Stevens Jr, Don L and Olsen, Anthony R},
journal={Journal of the American statistical Association},
volume={99},
number={465},
pages={262--278},
year={2004},
publisher={Taylor \& Francis}
}
