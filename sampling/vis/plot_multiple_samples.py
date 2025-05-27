#%%
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import rasterio
import cartopy.crs as ccrs
from pathlib import Path
import pandas as pd
from sampling.io.input_image import get_input_image

from config import LABEL_BAND_NAME
from sampling.vis.vis_params import (
    OSM_WATER_CMAP, LABEL_CMAP, SAMPLE_COLORS, WATER_COLOR, LAND_COLOR,
    TITLE_SIZE, AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_TITLE_SIZE, LEGEND_TEXT_SIZE, 
    LEGEND_OUTLINE_WIDTH, SUBPLOT_LEGEND_TEXT_SIZE, LINE_THICKNESS
)
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.lines as mlines
from sampling.vis.vis_params import *
from sampling.vis.utils import get_height_ratios_with_legend

# Handle relative imports for utils
try:
    from .utils import add_grid_lines, get_square_extent, get_data_extent
except ImportError:
    try:
        from sampling.vis.utils import add_grid_lines, get_square_extent, get_data_extent
    except ImportError:
        # Define fallback functions
        def get_data_extent(bounds):
            """Get data extent from bounds"""
            return [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        def get_square_extent(bounds):
            """Get square extent centered on data"""
            center_x = (bounds.left + bounds.right) / 2
            center_y = (bounds.bottom + bounds.top) / 2
            
            # Calculate the maximum dimension
            width = bounds.right - bounds.left
            height = bounds.top - bounds.bottom
            max_dim = max(width, height)
            
            # Create square extent
            half_dim = max_dim / 2
            return [
                center_x - half_dim,  # left
                center_x + half_dim,  # right
                center_y - half_dim,  # bottom
                center_y + half_dim   # top
            ]
        
        def add_grid_lines(*args, **kwargs):
            """Placeholder function"""
            pass

import math
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

# Handle relative import for vis_params
try:
    from .vis_params import DPI
except ImportError:
    try:
        from sampling.vis.vis_params import DPI
    except ImportError:
        # Fallback DPI value
        DPI = 300

def plot_multiple_samples(input_image_path, sample_paths, stats_path=None, custom_extent=None, show_lat_lon=True, reduced_legend=False, grid_paths=None, output_path=None, title_font_size=None, text_font_size=None, ncols=2, figsize=(12, 12)):
    """
    Plot multiple samples in subplots with the same base layers.
    
    Parameters:
    -----------
    input_image_path : str
        Path to the input GeoTIFF image
    sample_paths : list
        List of paths to sample point GeoJSON files
    stats_path : str, optional
        Path to the CSV file containing statistics for the sample sets
    custom_extent : None, 'square', or list, optional
        Controls the map extent:
        - None: use original data extent
        - 'square': use square extent centered on data
        - list [left, right, bottom, top]: use specified extent in degrees
    grid_paths : list or None, optional
        List of paths to GRTS grid GeoJSON files (same length as sample_paths)
        Each element can be None, a single path, or a tuple of two paths for two grids
    output_path : str, optional
        Path to save the output figure
    ncols : int, optional
        Number of columns in the subplot grid
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    
    """
    if title_font_size:
        title_font_size = title_font_size
    else:
        title_font_size = TITLE_SIZE
    
    if text_font_size:
        axis_label_size = text_font_size
        tick_label_size = text_font_size
        legend_text_size = text_font_size
        legend_title_size = text_font_size
        subplot_legend_text_size = text_font_size
    else:
        axis_label_size = AXIS_LABEL_SIZE
        tick_label_size = TICK_LABEL_SIZE
        legend_text_size = LEGEND_TEXT_SIZE
        legend_title_size = LEGEND_TITLE_SIZE
        subplot_legend_text_size = SUBPLOT_LEGEND_TEXT_SIZE
    # Validate inputs
    if not sample_paths:
        raise ValueError("At least one sample path must be provided")
    
    if grid_paths is not None and len(grid_paths) != len(sample_paths):
        raise ValueError("If grid_paths is provided, it must have the same length as sample_paths")
    
    # Read stats CSV if provided
    stats_df = None
    if stats_path:
        try:
            stats_df = pd.read_csv(stats_path)
        except Exception as e:
            print(f"Warning: Could not read stats file {stats_path}: {e}")
    
    # Read the input image
    image, metadata, label_band, osm_water_band = get_input_image(input_image_path)
    
    # Get projection and bounds information
    with rasterio.open(input_image_path) as src:
        src_crs = src.crs
        bounds = src.bounds
    
    # Get data extent (original bounds)
    data_extent = get_data_extent(bounds)
    
    # Determine which extent to use based on custom_extent parameter
    if custom_extent is None:
        # Use original data extent
        map_extent = data_extent
        print(f"Using original data extent: {map_extent}")
    elif custom_extent == 'square':
        # Use square extent centered on data
        map_extent = get_square_extent(bounds)
        print(f"Using square extent: {map_extent}")
    elif isinstance(custom_extent, (list, tuple)) and len(custom_extent) == 4:
        # Use user-specified extent: [left, right, bottom, top]
        map_extent = custom_extent
        print(f"Using custom extent: {map_extent}")
    else:
        # Invalid custom_extent value, fall back to data extent
        map_extent = data_extent
        print(f"Invalid custom_extent '{custom_extent}', using data extent: {map_extent}")
    
    # Define projections 
    # EPSG:4326 = WGS84 (standard lat/lon)
    data_crs = ccrs.PlateCarree()  # Our data is in EPSG:4326
    plot_crs = ccrs.PlateCarree()  # We'll use PlateCarree projection for the plot too

    # Calculate extent dimensions in kilometers
    lon_diff = abs(map_extent[1] - map_extent[0])  # width in degrees
    lat_diff = abs(map_extent[3] - map_extent[2])  # height in degrees
    
    # Calculate midpoint latitude for more accurate conversion
    mid_lat = (map_extent[2] + map_extent[3]) / 2
    
    # Convert degrees to kilometers
    # 1 degree of latitude is approximately 111 km
    # 1 degree of longitude varies with latitude
    km_per_lon_degree = 111 * math.cos(math.radians(mid_lat))
    km_per_lat_degree = 111
    
    # Calculate dimensions in km
    width_km = lon_diff * km_per_lon_degree
    height_km = lat_diff * km_per_lat_degree
    
    # Format the width dimension for display
    if width_km < 1:
        width_m = width_km * 1000
        extent_dimensions = f"{width_m:.0f} m"
    elif width_km < 10:
        extent_dimensions = f"{width_km:.1f} km" 
    else:
        extent_dimensions = f"{width_km:.0f} km"
    
    # Calculate layout
    n_cols_strategies = ncols  # Use the provided ncols parameter
    n_rows_strategies = int(np.ceil(len(sample_paths) / n_cols_strategies))

    width_ratios = [1] * n_cols_strategies
    height_ratios = [1] * n_rows_strategies 

    n_rows_with_legend, height_ratios_with_legend = get_height_ratios_with_legend(figsize[1], height_ratios=height_ratios)


    # Create figure with gridspec for better control
    fig = plt.figure(figsize=figsize)
    
    # Create gridspec with proper spacing
    gs = fig.add_gridspec(n_rows_with_legend, n_cols_strategies, width_ratios=width_ratios, height_ratios=height_ratios_with_legend)
    # Set line thickness for all subplots
    plt.rcParams['axes.linewidth'] = LINE_THICKNESS  # Outline (spine) thickness
    plt.rcParams['xtick.major.width'] = LINE_THICKNESS
    plt.rcParams['ytick.major.width'] = LINE_THICKNESS
    plt.rcParams['axes.edgecolor'] = LINE_COLOR  # Set spine color

    # Set default font sizes
    plt.rc('font', size=tick_label_size)          # controls default text sizes
    plt.rc('axes', titlesize=title_font_size)          # fontsize of the axes title
    plt.rc('axes', labelsize=axis_label_size)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=tick_label_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=tick_label_size)    # fontsize of the tick labels
    plt.rc('legend', fontsize=legend_text_size)   # legend fontsize
    plt.rc('figure', titlesize=title_font_size)        # fontsize of the figure title
    # Prepare legend items - organize by categories for column layout
    sample_legend_items = [
        mlines.Line2D([], [], color=SAMPLE_COLORS['1'], marker='+', markeredgewidth=3, linestyle='None', markersize=15, label='Flooded Sample'),
        mlines.Line2D([], [], color=SAMPLE_COLORS['0'], marker='+', markeredgewidth=3, linestyle='None', markersize=15, label='Non-Flooded Sample')
    ]
    
    stratum_legend_items = [
        mpatches.Patch(color=LABEL_CMAP(1), label='Flooded Stratum'),
        mpatches.Patch(color=LABEL_CMAP(0), label='Non-Flooded Stratum'),
        mpatches.Patch(color=OSM_WATER_CMAP(1), label='OSM Water')
    ]
    
    
    # Define colors and labels for different grid types
    grid_colors = {
        'flooded': WATER_COLOR,
        'non_flooded': LAND_COLOR,
        'simple': 'black'
    }
    
    # Keep track of grid legends to avoid duplicates
    grid_legends = {}
    
    # Plot each sample in its own subplot
    for i, sample_path in enumerate(sample_paths):
        # Create subplot with specified projection using gridspec
        row = i // n_cols_strategies
        col = i % n_cols_strategies
        ax = fig.add_subplot(gs[row, col], projection=plot_crs)
        
        # Get sample title from file path
        if isinstance(sample_path, str):
            title = Path(sample_path).stem
        else:
            title = sample_path.stem
            
        # Clean up the title: remove underscores and 'samples' word
        title = title.replace('_', ' ')
        title = title.replace('samples', '').replace('sample', '')
        title = title.strip()
        title = ' '.join(word.upper() if word.lower() == 'grts' else word.capitalize() for word in title.split())
        #title = '\n'.join(title.split())  # Add line breaks between words
        ax.set_title(title, fontsize=TITLE_SIZE, weight='bold', pad=10)


        # Plot base layers (these are raster arrays with bounds in EPSG:4326)
        ax.imshow(label_band, cmap=LABEL_CMAP, alpha=1,
                extent=data_extent, transform=data_crs, origin='upper')
        
        ax.imshow(osm_water_band, cmap=OSM_WATER_CMAP, alpha=1, 
                extent=data_extent, transform=data_crs, origin='upper')
        
        # Read and plot sample points
        sample_counts = {0: 0, 1: 0}  # Count of samples by label
        min_spacing = None
        
        # Get filename for matching with stats CSV
        sample_filename = Path(sample_path).name
        
        # Try to get sample counts and min_spacing from stats CSV
        if stats_df is not None:
            try:
                # Find the matching row in the stats dataframe
                matching_row = stats_df[stats_df['file_path'] == sample_filename]
                if not matching_row.empty:
                    # Get flooded and non-flooded counts
                    sample_counts[1] = matching_row['flooded'].values[0]
                    sample_counts[0] = matching_row['non_flooded'].values[0]
                    # Get min_spacing
                    min_spacing = matching_row['min_spacing'].values[0]
                    # No need to count samples from file
                    samples_counted = True
                else:
                    samples_counted = False
            except Exception as e:
                print(f"Warning: Error reading stats for {sample_filename}: {e}")
                samples_counted = False
        else:
            samples_counted = False
        
        # If we couldn't get stats from CSV, count samples from file
        if not samples_counted:
            try:
                samples = gpd.read_file(sample_path)
                
                # Count samples in each category
                for strata in ['0', '1']:
                    try:
                        strata_samples = samples[samples[LABEL_BAND_NAME] == int(strata)]
                        sample_counts[int(strata)] = len(strata_samples)
                        if not strata_samples.empty:
                            # Use GeoPandas plot with the correct transform
                            strata_samples.plot(
                                ax=ax, color=SAMPLE_COLORS[strata], markersize=100, 
                                transform=data_crs,
                                marker='+',
                            )
                    except Exception as e:
                        print(f"Warning: Could not plot stratum {strata} for sample {title}: {e}")
            except Exception as e:
                print(f"Warning: Could not read sample file {sample_path}: {e}")
        else:
            # Still need to plot the samples even if we got counts from CSV
            try:
                samples = gpd.read_file(sample_path)
                
                # Plot samples in each category
                for strata in ['0', '1']:
                    try:
                        strata_samples = samples[samples[LABEL_BAND_NAME] == int(strata)]
                        if not strata_samples.empty:
                            # Use GeoPandas plot with the correct transform
                            strata_samples.plot(
                                ax=ax, color=SAMPLE_COLORS[strata], markersize=100, 
                                transform=data_crs,
                                marker='+',
                            )
                    except Exception as e:
                        print(f"Warning: Could not plot stratum {strata} for sample {title}: {e}")
            except Exception as e:
                print(f"Warning: Could not read sample file {sample_path}: {e}")
        
        # Plot grid(s) if provided
        if grid_paths is not None and i < len(grid_paths) and grid_paths[i] is not None:
            # Handle both single grid and tuple of two grids
            grid_items = []
            if isinstance(grid_paths[i], tuple) and len(grid_paths[i]) == 2:
                # Identify grid types from filename
                grid_path1, grid_path2 = grid_paths[i]
                
                # Try to determine grid type from filename
                if grid_path1 is not None:
                    path_str = str(grid_path1).lower()
                    if 'flood' in path_str:
                        color1 = grid_colors['flooded']
                        label1 = 'Flooded Grid'
                    elif 'non' in path_str:
                        color1 = grid_colors['non_flooded']
                        label1 = 'Non-Flooded Grid'
                    else:
                        color1 = grid_colors['simple']
                        label1 = 'Simple Grid'
                    grid_items.append((grid_path1, color1, label1))
                
                if grid_path2 is not None:
                    path_str = str(grid_path2).lower()
                    if 'flood' in path_str and 'non' not in path_str:
                        color2 = grid_colors['flooded']
                        label2 = 'Flooded Grid'
                    elif 'non' in path_str:
                        color2 = grid_colors['non_flooded']
                        label2 = 'Non-Flooded Grid'
                    else:
                        color2 = grid_colors['simple']
                        label2 = 'Simple Grid'
                    grid_items.append((grid_path2, color2, label2))
            else:
                # Single grid - determine type from filename
                grid_path = grid_paths[i]
                path_str = str(grid_path).lower()
                
                if 'flood' in path_str and 'non' not in path_str:
                    color = grid_colors['flooded']
                    label = 'Flooded Grid'
                elif 'non' in path_str:
                    color = grid_colors['non_flooded'] 
                    label = 'Non-Flooded Grid'
                else:
                    color = grid_colors['simple']
                    label = 'Simple Grid'
                
                grid_items.append((grid_path, color, label))
            
            # Plot each grid
            for grid_path, color, label in grid_items:
                try:
                    grid = gpd.read_file(grid_path)
                    
                    # Use Cartopy's native method for plotting grid
                    for geom in grid.geometry:
                        if hasattr(geom, 'boundary'):
                            x, y = geom.boundary.xy
                            ax.plot(x, y, color=color, linewidth=1, transform=data_crs)
                    
                    # Add to legend if not already there
                    if label not in grid_legends:
                        grid_legends[label] = mlines.Line2D([], [], color=color, linewidth=1, label=label)
                except Exception as e:
                    print(f"Warning: Could not plot grid {grid_path}: {e}")
        
        # Set map extent for all subplots
        ax.set_extent(map_extent, crs=data_crs)
        
        # Set aspect ratio for all subplots
        ax.set_aspect('equal')
        
        # Set Linethickness for all subplots
        plt.rcParams['axes.linewidth'] = LINE_THICKNESS  # Outline (spine) thickness
        plt.rcParams['xtick.major.width'] = LINE_THICKNESS
        plt.rcParams['ytick.major.width'] = LINE_THICKNESS

        # Only add lat/lon labels to the last subplot
        if i == len(sample_paths) - 1 and show_lat_lon:
            gl = ax.gridlines(draw_labels=True, linewidth=0, linestyle='', alpha=0)
            gl.top_labels = False
            gl.left_labels = False
            gl.xformatter = LongitudeFormatter(number_format='.1f')
            gl.yformatter = LatitudeFormatter(number_format='.1f')
            gl.xlocator = mticker.LinearLocator(3) 
            gl.ylocator = mticker.LinearLocator(3)  
            
            # Increase font size and rotate labels
            gl.xlabel_style = {'size': AXIS_LABEL_SIZE}
            gl.ylabel_style = {'size': AXIS_LABEL_SIZE, 'rotation': 270}
        else:
            # No gridlines for all subplots
            ax.gridlines(draw_labels=False, linewidth=0, linestyle='', alpha=0)
        
        # Create individual legend for this subplot showing sample counts and proportions
        total_samples = sample_counts[0] + sample_counts[1]
        
        if total_samples > 0:
            # Create legend items list
            legend_items = []
            
            # Create simple text-only legend entries
            f_legend = mlines.Line2D([], [], color='none', marker=None, linestyle='None', 
                                    label=f'nF: {sample_counts[1]}')
            legend_items.append(f_legend)
            
            nf_legend = mlines.Line2D([], [], color='none', marker=None, linestyle='None', 
                                     label=f'nNF: {sample_counts[0]}')
            legend_items.append(nf_legend)
            
            # Add min_spacing if available
            if min_spacing is not None:
                # Format min_spacing in meters with appropriate precision
                if min_spacing < 1000:  # Less than 1km
                    dmin_formatted = f"{min_spacing:.0f} m"
                elif min_spacing < 10000:  # Less than 10km
                    dmin_formatted = f"{(min_spacing/1000):.1f} km"
                else:  # 10km or more
                    dmin_formatted = f"{(min_spacing/1000):.0f} km"
                
                dmin_legend = mlines.Line2D([], [], color='none', marker=None, linestyle='None',
                                           label=f"dMIN: {dmin_formatted}")
                legend_items.append(dmin_legend)
            
            # Create a simple text-only legend at the bottom left
            subplot_legend = ax.legend(
                handles=legend_items,
                loc='lower left',
                bbox_to_anchor=(0, 0),
                fontsize=subplot_legend_text_size,
                frameon=False,  # No frame
                ncol=1,
                borderpad=0.02,
                handletextpad=0,
                handlelength=0,
                labelspacing=0.1
            )
            
            # Add the legend to the subplot
            ax.add_artist(subplot_legend)
    
    # Organize grid legends
    grid_legend_items = list(grid_legends.values())
    
    # Organize legend elements into columns
    legend_columns = [
        sample_legend_items,       # Column 1: Sample types
        stratum_legend_items,      # Column 2: Stratum types 
        grid_legend_items          # Column 3: Grid types
    ]
    
    if not reduced_legend:
        # Create extent dimensions and dMIN legend items
        extent_legend = mlines.Line2D([], [], color='none', marker=None, linestyle='none',
                               label=f"Subplot Width: {extent_dimensions}", markersize=0)
        n_legend = mlines.Line2D([], [], color='none', marker=None, linestyle='none',
                                label=f"nF, nNF: sample counts", markersize=0)
        dmin_legend = mlines.Line2D([], [], color='none', marker=None, linestyle='none',
                                label=f"dMin: min sample spacing", markersize=0)

        # Add extent dimensions and dMIN to legend elements
        extent_legend_items = [extent_legend, n_legend, dmin_legend]
        legend_columns.append(extent_legend_items)
    
    # Combine all legend elements
    legend_elements = []
    for column in legend_columns:
        legend_elements.extend(column)
    
    legend_ax = fig.add_subplot(gs[n_rows_strategies, :])
    legend_ax.axis('off')
    #legend_ax.set_xticks([])
    #legend_ax.set_yticks([])

    # Add a single legend at the bottom of the figure with column organization
    legend = legend_ax.legend(handles=legend_elements, loc='center', 
                 bbox_to_anchor=(0.5, 0.5),  # Position within bottom margin
                 ncol=len(legend_columns),     # Adjust number of columns based on content
                 fontsize=legend_text_size,
                 frameon=False, 
                 facecolor='none',
                 #borderpad=0,
                 labelspacing=0.5, 
                 #handleheight=0.5,
                 #borderaxespad=0,
                 )



    plt.tight_layout(pad=0.5)
    # Remove invalid subplots_adjust call - not needed with gridspec
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if output_path:
        # Save figure
        plt.savefig(output_path, dpi=DPI, transparent=True)
    
    return fig 

#%%
if __name__ == "__main__":
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
        samples_dir / 'locations' / 'balanced_random.geojson',
        samples_dir / 'locations' / 'simple_systematic.geojson',
        samples_dir / 'locations' / 'balanced_systematic.geojson',
    ]



    grid_paths = None # comment out to use grid_paths above
    fig = plot_multiple_samples(
        input_image_path=input_image_path,
        sample_paths=sample_paths,
        stats_path=samples_dir / 'locations' / 'stats.csv',
        custom_extent='square',
        #grid_paths=grid_paths, # uncomment to show grids
        ncols=4,
        show_lat_lon=False,
        figsize=(12, 4),
        reduced_legend=True,
        output_path=plot_dir / f'select_sample_sets_{iteration_id}_{site_id}_{n_samples}.png'
    ) 
# %%
