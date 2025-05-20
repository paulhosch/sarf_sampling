# %% Imports
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from sampling.vis.plot_sample_hist import plot_sample_histograms

# %% Example usage
if __name__ == "__main__":
    categorical_mapping = {
        "LAND_COVER": {
            1: "W", # Water
            2: "T", # Trees
            4: "FV", # Flooded Vegetation
            5: "C", # Crops
            7: "BA", # Built Area
            8: "BG", # Bare Ground
            9: "SI", # Snow and Ice
            10: "C", # Clouds
            11: "R" # Rangeland
        }
    }

    select_site = "valencia"
    select_strategy = "simple_random"
    select_sample_size = 1000
    select_iteration = "iteration_1"  # Specify a particular iteration

    fig_size = (16, 10)
    n_bins = 30
    output_dir = f"../../data/case_studies/{select_site}/samples/plots"
    os.makedirs(output_dir, exist_ok=True)

    features_to_plot = [
        # SAR features
        "VV_POST", "VV_PRE", "VH_POST", "VH_PRE",
        "VV_CHANGE", "VH_CHANGE",
        "VV_VH_RATIO_PRE", "VV_VH_RATIO_POST", "VV_VH_RATIO_CHANGE",
        # Contextual features
        "SLOPE", "LAND_COVER", "HAND", "DEM", "EDTW", 
        # Label
        "LABEL"
    ]

    # Example 1: Compare strategies for one site, sample size, and iteration
    fig = plot_sample_histograms(
        base_dir="../../data",
        compare="strategy",
        select_site=select_site,
        select_sample_size=select_sample_size,
        select_iteration=select_iteration,  # Specify which iteration to use
        features_to_plot=features_to_plot,
        categorical_features=categorical_mapping,
        binary_features=["LABEL"],
        title=f"Sample Histograms for {select_site} with {select_sample_size} samples (Iteration: {select_iteration})",
        output_path=os.path.join(output_dir, f"sample_histograms_{select_site}_{select_sample_size}_{select_iteration}.png"),
        figsize=fig_size,
        normalize="count",
        n_bins=n_bins  # Increase number of bins for finer resolution
    )
    
    # Example 2: Compare across iterations for one site, strategy, and sample size
    fig = plot_sample_histograms(
        base_dir="../../data",
        compare="iterations",  # New comparison option
        select_site=select_site,
        select_strategy=select_strategy,
        select_sample_size=select_sample_size,
        features_to_plot=features_to_plot,
        categorical_features=categorical_mapping,
        binary_features=["LABEL"],
        title=f"Comparing Iterations for {select_site}, {select_strategy}, {select_sample_size} samples",
        output_path=os.path.join(output_dir, f"iteration_comparison_{select_site}_{select_strategy}_{select_sample_size}.png"),
        figsize=fig_size,
        normalize="count",
        n_bins=n_bins
    )
    
    # Example 3: Compare sample sizes for one site and strategy (original examples)
    fig = plot_sample_histograms(
        base_dir="../../data",
        compare="sample_size",
        select_site=select_site,
        select_strategy=select_strategy,
        select_iteration=select_iteration,  # Can specify iteration if needed
        features_to_plot=features_to_plot,
        categorical_features=categorical_mapping,
        binary_features=["LABEL"],
        title=f"Sample Histograms for {select_site}, {select_strategy} (compare sample sizes)",
        output_path=os.path.join(output_dir, f"sample_histograms_{select_site}_{select_strategy}.png"),
        figsize=fig_size,
        normalize="count",
        n_bins=n_bins
    )
    
    # Example 4: Compare sites for one strategy and sample size
    fig = plot_sample_histograms(
        base_dir="../../data",
        compare="site",
        select_strategy=select_strategy,
        select_sample_size=select_sample_size,
        select_iteration=select_iteration,  # Can specify iteration if needed
        features_to_plot=features_to_plot,
        categorical_features=categorical_mapping,
        binary_features=["LABEL"],
        title=f"Sample Histograms for {select_strategy}, {select_sample_size} samples (compare sites)",
        output_path=os.path.join(output_dir, f"sample_histograms_{select_strategy}_{select_sample_size}.png"),
        figsize=fig_size,
        normalize="count",
        n_bins=n_bins
    )
# %%