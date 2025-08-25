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

    fig_size = (12.99, 4.72)
    n_bins = 50
    output_dir = f"../../data/case_studies/{select_site}/samples/plots"
    os.makedirs(output_dir, exist_ok=True)

    features_to_plot = [
        "DEM",
        "HAND"
    ]

    

    #%% Example 2: Compare across iterations for one site, strategy, and sample size
    
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
        #title=f"Sample Histograms for {select_strategy}, {select_sample_size} samples (compare sites)",
        output_path=os.path.join(output_dir, f"dem_v_hand_{select_strategy}_{select_sample_size}.png"),
        figsize=fig_size,
        #legend_height=1,
        normalize="count",
        n_bins=n_bins,
        #auto_size=False,
        use_value_label=True,
        legend_title="Study Site",
        custom_colors=['#CB78BB', '#009E73', '#0173B2'],
        legend_rows=2
    )
# %%