# %% Imports
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

# Import visualization parameters
from sampling.vis.vis_params import (
    TITLE_SIZE, 
    AXIS_LABEL_SIZE, 
    TICK_LABEL_SIZE, 
    LEGEND_TITLE_SIZE, 
    LEGEND_TEXT_SIZE, 
    SUBPLOT_LEGEND_TEXT_SIZE
)

# Add SUBPLOT_TITLE_SIZE if not present
try:
    from sampling.vis.vis_params import SUBPLOT_TITLE_SIZE
except ImportError:
    SUBPLOT_TITLE_SIZE = 16

# Define visualization parameters for features
# This is a copy of VIS_PARAMS from vis_params.py to avoid import issues
VIS_PARAMS = {
    "DEM": {
        "title": "Terrain Model",
        "value_label": "Elevation (m)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "BURNED_DEM": {
        "title": "Stream-Burned Terrain Model", 
        "value_label": "Elevation (m)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "SLOPE": {
        "title": "Horton's Slope",
        "value_label": "Slope (°)",
        "vmin": 0,
        "vmax": 90
    },
    "OSM_WATER": {
        "title": "Water Features",
        "value_label": "Water presence",
        "vmin": "auto",
        "vmax": "auto"
    },
    "EDTW": {
        "title": "Euclidean Distance to Water",
        "value_label": "Distance (m)", 
        "vmin": "auto",
        "vmax": "auto"
    },
    "HAND": {
        "title": "Height Above Nearest Drainage",
        "value_label": "HAND (m)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VV_PRE": {
        "title": "Pre-Event VV",
        "value_label": r"$\sigma^0$ (dB)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VH_PRE": {
        "title": "Pre-Event VH",
        "value_label": r"$\sigma^0$ (dB)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VV_POST": {
        "title": "Post-Event VV",
        "value_label": r"$\sigma^0$ (dB)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VH_POST": {
        "title": "Post-Event VH",
        "value_label": r"$\sigma^0$ (dB)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VV_CHANGE": {
        "title": "VV Change",
        "value_label": "Post/Pre (-)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VH_CHANGE": {
        "title": "VH Change",
        "value_label": "Post/Pre (-)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VV_VH_RATIO_PRE": {
        "title": "Pre-event VV/VH",
        "value_label": "Ratio (-)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VV_VH_RATIO_POST": {
        "title": "Post-event VV/VH",
        "value_label": "Ratio (-)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "VV_VH_RATIO_CHANGE": {
        "title": "VV/VH Change",
        "value_label": "Ratio (-)",
        "vmin": "auto",
        "vmax": "auto"
    },
    "LAND_COVER": {
        "title": "Land Cover",
        "value_label": "Class",
        "vmin": "auto",
        "vmax": "auto"
    },
    "LABEL": {
        "title": "Flood Label",
        "value_label": "Flood presence",
        "vmin": "auto",
        "vmax": "auto"
    }
}

# %% Plotting function
def plot_sample_histograms(
    base_dir: str = "../../data",
    compare: str = "strategy",  # 'site', 'strategy', 'sample_size', or 'iterations'
    select_site: Optional[str] = None,
    select_strategy: Optional[str] = None,
    select_sample_size: Optional[int] = None,
    select_iteration: Optional[str] = None,  # New parameter for filtering by iteration
    features_to_plot: List[str] = None,
    categorical_features: Dict[str, Dict[int, str]] = None,
    binary_features: List[str] = None,
    title: str = "Histogram of Sampled Values",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 300,
    n_bins: int = 30,
    subplot_adjust: Dict[str, float] = None,
    normalize: str = "count",  # Options: "count" or "density"
):
    """
    Plot histograms of sampled values, comparing one variable at a time using seaborn.
    
    Parameters:
    -----------
    base_dir : str
        Base directory for data
    compare : str
        What to compare: 'site', 'strategy', 'sample_size', or 'iterations'
    select_site : str, optional
        Filter to a specific site
    select_strategy : str, optional
        Filter to a specific strategy
    select_sample_size : int, optional
        Filter to a specific sample size
    select_iteration : str, optional
        Filter to a specific iteration (e.g., 'iteration_1')
    features_to_plot : list, optional
        List of features to plot
    categorical_features : dict, optional
        Dictionary mapping feature names to dictionaries mapping values to labels
    binary_features : list, optional
        List of binary features to plot as horizontal bar charts
    title : str
        Title for the plot
    output_path : str, optional
        Path to save the plot
    figsize : tuple
        Figure size (width, height)
    dpi : int
        DPI for saved figure
    n_bins : int
        Number of bins for histograms
    subplot_adjust : dict, optional
        Dictionary with subplot adjustment parameters
    normalize : str, optional
        How to normalize the histograms: "count" (default) or "density"
    """
    assert compare in ["site", "strategy", "sample_size", "iterations"], "compare must be one of 'site', 'strategy', 'sample_size', or 'iterations'"
    assert normalize in ["count", "density"], "normalize must be one of 'count' or 'density'"

    # Default categorical feature mappings if not provided
    if categorical_features is None:
        categorical_features = {
            "LAND_COVER": {
                0: "Urban",
                1: "Agriculture",
                2: "Forest",
                3: "Water",
                4: "Barren",
                5: "Other"
            }
        }
    if binary_features is None:
        binary_features = ["LABEL"]

    # Find all case study directories
    site_dirs = glob.glob(os.path.join(base_dir, "case_studies", "*"))
    all_sites = [os.path.basename(site_dir) for site_dir in site_dirs]
    
    # Filter sites if selected
    if select_site is not None:
        all_sites = [select_site]

    # Load all sample data
    all_data = []
    for site_id in all_sites:
        # Check for iteration folders first (new directory structure)
        iteration_dirs = glob.glob(os.path.join(base_dir, "case_studies", site_id, "samples", "iteration_*"))
        
        if iteration_dirs:
            # We have iteration-based structure
            iterations_to_process = []
            if select_iteration is not None:
                # Use specified iteration if it exists
                iteration_path = os.path.join(base_dir, "case_studies", site_id, "samples", select_iteration)
                if os.path.exists(iteration_path):
                    iterations_to_process = [select_iteration]
            elif compare == "iterations":
                # Process all iterations for comparison
                iterations_to_process = [os.path.basename(d) for d in iteration_dirs]
            else:
                # Default to first iteration
                iterations_to_process = [os.path.basename(iteration_dirs[0])]
            
            for iteration in iterations_to_process:
                iteration_path = os.path.join(base_dir, "case_studies", site_id, "samples", iteration)
                
                # Get sample sizes
                size_dirs = glob.glob(os.path.join(iteration_path, "training", "*"))
                curr_sample_sizes = []
                for size_dir in size_dirs:
                    try:
                        curr_sample_sizes.append(int(os.path.basename(size_dir)))
                    except ValueError:
                        continue
                
                # Filter by selected sample size if specified
                if select_sample_size is not None:
                    curr_sample_sizes = [n for n in curr_sample_sizes if n == select_sample_size]
                
                for n_samples in curr_sample_sizes:
                    sample_dir = os.path.join(iteration_path, "training", str(n_samples), "samples")
                    if not os.path.exists(sample_dir):
                        continue
                    
                    strategy_files = glob.glob(os.path.join(sample_dir, "*.csv"))
                    for f in strategy_files:
                        strategy = os.path.splitext(os.path.basename(f))[0]
                        
                        # Skip if not selected strategy
                        if select_strategy is not None and strategy != select_strategy:
                            continue
                        
                        try:
                            df = pd.read_csv(f)
                            df["site_id"] = site_id
                            df["n_samples"] = n_samples
                            df["strategy"] = strategy
                            df["iteration"] = iteration
                            all_data.append(df)
                        except Exception as e:
                            print(f"Warning: Failed to load {f}: {e}")
        else:
            # Try original directory structure without iterations
            size_dirs = glob.glob(os.path.join(base_dir, "case_studies", site_id, "samples", "training", "*"))
            if not size_dirs:
                size_dirs = glob.glob(os.path.join(base_dir, "case_studies", site_id, "samples", "*"))
            
            curr_sample_sizes = []
            for size_dir in size_dirs:
                try:
                    curr_sample_sizes.append(int(os.path.basename(size_dir)))
                except ValueError:
                    continue
            
            # Filter by selected sample size if specified
            if select_sample_size is not None:
                curr_sample_sizes = [n for n in curr_sample_sizes if n == select_sample_size]
            
            for n_samples in curr_sample_sizes:
                sample_path = os.path.join(base_dir, "case_studies", site_id, "samples", "training", str(n_samples))
                if not os.path.exists(sample_path):
                    sample_path = os.path.join(base_dir, "case_studies", site_id, "samples", str(n_samples))
                
                sample_dir = os.path.join(sample_path, "samples")
                if not os.path.exists(sample_dir):
                    continue
                
                strategy_files = glob.glob(os.path.join(sample_dir, "*.csv"))
                for f in strategy_files:
                    strategy = os.path.splitext(os.path.basename(f))[0]
                    
                    # Skip if not selected strategy
                    if select_strategy is not None and strategy != select_strategy:
                        continue
                    
                    try:
                        df = pd.read_csv(f)
                        df["site_id"] = site_id
                        df["n_samples"] = n_samples
                        df["strategy"] = strategy
                        df["iteration"] = "original"  # Mark as original structure
                        all_data.append(df)
                    except Exception as e:
                        print(f"Warning: Failed to load {f}: {e}")
    
    if not all_data:
        raise ValueError("No sample data found with the specified criteria")
    
    combined_df = pd.concat(all_data, ignore_index=True)

    # Map categorical values to class names
    for cat_feat, mapping in categorical_features.items():
        if cat_feat in combined_df.columns:
            combined_df[cat_feat] = combined_df[cat_feat].map(mapping).fillna(combined_df[cat_feat])

    # Determine features to plot
    if features_to_plot is None:
        features_to_plot = [col for col in combined_df.columns if col not in ["site_id", "n_samples", "strategy", "iteration", "x", "y"]]

    # Filter by selected values
    df = combined_df.copy()
    if select_site is not None:
        df = df[df["site_id"] == select_site]
    if select_strategy is not None and compare != "strategy":
        df = df[df["strategy"] == select_strategy]
    if select_sample_size is not None and compare != "sample_size":
        df = df[df["n_samples"] == select_sample_size]
    if select_iteration is not None and compare != "iterations":
        df = df[df["iteration"] == select_iteration]

    # Set up seaborn color palette
    base_palette = sns.color_palette("colorblind")

    # Set up subplots
    n_features = len(features_to_plot)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    row_height = 4
    legend_height = 1
    title_height = 0.7
    fig_height = n_rows * row_height + legend_height + title_height
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], fig_height))
    if n_features == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # What to compare
    compare_col = {
        "site": "site_id",
        "strategy": "strategy",
        "sample_size": "n_samples",
        "iterations": "iteration"
    }[compare]

    handles_labels = []
    legend_labels = None
    legend_colors = None
    alpha_bar = 0.6  # Same as for histograms
    
    # Set y-axis label based on normalization
    y_label = "Density" if normalize == "density" else "Frequency"
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        # Get unique groups for this feature
        if feature in categorical_features or feature in binary_features:
            plot_df = df[[feature, compare_col]].copy()
            unique_groups = plot_df[compare_col].unique()
        else:
            unique_groups = df[compare_col].unique()
        palette = base_palette[:len(unique_groups)]
        
        # Get feature visualization parameters
        feature_params = VIS_PARAMS.get(feature, {})
        feature_title = feature_params.get("title", feature)
        value_label = feature_params.get("value_label", "Value")
        vmin = feature_params.get("vmin", None)
        vmax = feature_params.get("vmax", None)
        
        if feature in categorical_features:
            plot_df[feature] = plot_df[feature].astype(str)
            plot_df["count"] = 1
            
            if normalize == "density":
                # For categorical features with density normalization, we need to manually normalize
                normalized_data = []
                for group in unique_groups:
                    group_data = plot_df[plot_df[compare_col] == group]
                    value_counts = group_data[feature].value_counts()
                    total = value_counts.sum()
                    
                    for val, count in value_counts.items():
                        normalized_data.append({
                            feature: val,
                            compare_col: group,
                            "value": count / total if total > 0 else 0
                        })
                
                normalized_df = pd.DataFrame(normalized_data)
                g = sns.barplot(
                    data=normalized_df,
                    x=feature,
                    y="value",
                    hue=compare_col,
                    palette=palette,
                    ax=ax,
                    alpha=alpha_bar
                )
            else:
                # Standard count plot for raw frequencies
                g = sns.countplot(
                    data=plot_df,
                    x=feature,
                    hue=compare_col,
                    palette=palette,
                    ax=ax,
                    dodge=True,
                    alpha=alpha_bar
                )
            
            ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
            ax.set_xlabel("")
            # Rotate x-axis labels for categorical features
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
        elif feature in binary_features:
            plot_df[feature] = plot_df[feature].astype(str)
            plot_df["count"] = 1
            
            if normalize == "density":
                # For binary features with density normalization
                normalized_data = []
                for group in unique_groups:
                    group_data = plot_df[plot_df[compare_col] == group]
                    value_counts = group_data[feature].value_counts()
                    total = value_counts.sum()
                    
                    for val, count in value_counts.items():
                        normalized_data.append({
                            feature: val,
                            compare_col: group,
                            "value": count / total if total > 0 else 0
                        })
                
                normalized_df = pd.DataFrame(normalized_data)
                g = sns.barplot(
                    data=normalized_df,
                    y=feature,
                    x="value",
                    hue=compare_col,
                    palette=palette,
                    ax=ax,
                    alpha=alpha_bar
                )
            else:
                # Standard countplot for raw frequencies
                g = sns.countplot(
                    data=plot_df,
                    y=feature,
                    hue=compare_col,
                    palette=palette,
                    ax=ax,
                    dodge=True,
                    alpha=alpha_bar
                )
            
            ax.set_xlabel(y_label, fontsize=AXIS_LABEL_SIZE)
            ax.set_ylabel("")
            
        else:
            # For continuous features, calculate appropriate limits
            if vmin == 'auto':
                vmin = df[feature].quantile(0.05)  # 5th percentile
            if vmax == 'auto':
                vmax = df[feature].quantile(0.95)  # 95th percentile
                
            # Calculate bins based on vmin and vmax
            feature_bins = np.linspace(vmin, vmax, n_bins + 1)
            
            # Seaborn's histplot supports normalization directly
            g = sns.histplot(
                data=df,
                x=feature,
                hue=compare_col,
                palette=palette,
                ax=ax,
                bins=feature_bins,  # Use generated bins within the specified range
                element="step",
                stat=normalize,  # Use the normalize parameter directly
                common_norm=False,
                fill=True,
                linewidth=1.5,
                alpha=alpha_bar
            )
            
            # Set the limits explicitly to match the bin range
            ax.set_xlim([vmin, vmax])
            
            ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
            ax.set_xlabel("")
            
        # Use the feature title from VIS_PARAMS instead of the raw feature name
        ax.set_title(feature_title, fontsize=SUBPLOT_TITLE_SIZE, fontweight="bold")
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
        # Remove legend from axes
        ax.get_legend().remove()
        # Save legend labels/colors from the first plot
        if i == 0:
            legend_labels = list(unique_groups)
            legend_colors = palette

    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    if subplot_adjust:
        plt.subplots_adjust(**subplot_adjust)
    else:
        plt.tight_layout()
    plt.suptitle(title, fontsize=TITLE_SIZE, y=1.0)
    # Add combined legend below all subplots
    if legend_labels is not None and legend_colors is not None:
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=col, edgecolor='k', label=str(lab), alpha=alpha_bar) for col, lab in zip(legend_colors, legend_labels)]
        fig.legend(
            handles=legend_handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.04),
            ncol=min(len(legend_handles), n_cols),
            fontsize=LEGEND_TEXT_SIZE,
            frameon=False,
            title=compare_col,
            title_fontsize=LEGEND_TITLE_SIZE
        )
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    return fig

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

    fig_size = (16, 10)
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

    # Compare all strategies for one site and sample size
    fig = plot_sample_histograms(
        base_dir="../../data",
        compare="strategy",
        select_site=select_site,
        select_sample_size=select_sample_size,
        features_to_plot=features_to_plot,
        categorical_features=categorical_mapping,
        binary_features=["LABEL"],
        title=f"Sample Histograms for {select_site} with {select_sample_size} samples (compare strategies)",
        output_path=os.path.join(output_dir, f"{select_site}_{select_sample_size}_sample_histograms.png"),
        figsize=fig_size,
    )
    # Compare all sample sizes for one site and strategy
    fig = plot_sample_histograms(
        base_dir="../../data",
        compare="sample_size",
        select_site=select_site,
        select_strategy=select_strategy,
        features_to_plot=features_to_plot,
        categorical_features=categorical_mapping,
        binary_features=["LABEL"],
        title=f"Sample Histograms for {select_site}, {select_strategy} (compare sample sizes)",
        output_path=os.path.join(output_dir, f"{select_site}_{select_strategy}_sample_histograms.png"),
        figsize=fig_size,
    )
    # Compare all sites for one strategy and sample size
    fig = plot_sample_histograms(
        base_dir="../../data",
        compare="site",
        select_strategy=select_strategy,
        select_sample_size=select_sample_size,
        features_to_plot=features_to_plot,
        categorical_features=categorical_mapping,
        binary_features=["LABEL"],
        title=f"Sample Histograms for {select_strategy}, {select_sample_size} samples (compare sites)",
        output_path=os.path.join(output_dir, f"{select_strategy}_{select_sample_size}_sample_histograms.png"),
        figsize=fig_size,
    )
# %%
