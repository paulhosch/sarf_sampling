import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import rasterio
import cartopy.crs as ccrs
from pathlib import Path
from ..io.input_image import get_input_image
from config import LABEL_BAND_NAME
from sampling.vis.vis_params import (
    OSM_WATER_CMAP, LABEL_CMAP, SAMPLE_COLORS, WATER_COLOR, LAND_COLOR,
    TITLE_SIZE, AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_TITLE_SIZE, LEGEND_TEXT_SIZE, 
    LEGEND_OUTLINE_WIDTH, SUBPLOT_LEGEND_TEXT_SIZE
)
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.lines as mlines
from .utils import add_grid_lines, get_square_extent, get_data_extent
import math

def plot_multiple_samples(input_image_path, sample_paths, custom_extent=None, grid_paths=None, output_path=None, ncols=2, figsize=(12, 12)):
    """
    Plot multiple samples in subplots with the same base layers.
    
    Parameters:
    -----------
    input_image_path : str
        Path to the input GeoTIFF image
    sample_paths : list
        List of paths to sample point GeoJSON files
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
    # Validate inputs
    if not sample_paths:
        raise ValueError("At least one sample path must be provided")
    
    if grid_paths is not None and len(grid_paths) != len(sample_paths):
        raise ValueError("If grid_paths is provided, it must have the same length as sample_paths")
    
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

    # Calculate layout
    n_samples = len(sample_paths)
    nrows = math.ceil(n_samples / ncols)
    
    # Create figure with cartopy
    fig = plt.figure(figsize=figsize)
    
    # Set default font sizes
    plt.rc('font', size=TICK_LABEL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=TITLE_SIZE)          # fontsize of the axes title
    plt.rc('axes', labelsize=AXIS_LABEL_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TICK_LABEL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TICK_LABEL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=LEGEND_TEXT_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=TITLE_SIZE)        # fontsize of the figure title
    
    # Add spacing parameters for the subplots
    plt.subplots_adjust(
        wspace=0,    # width spacing between subplots
        hspace=0.2,    # height spacing between subplots
        top=1,      # top boundary (more space below title)
        bottom=0.2,   # bottom boundary (space for legend)
        left=0,     # left boundary
        right=1    # right boundary
    )
    
    # Prepare legend items - organize by categories for column layout
    sample_legend_items = [
        mlines.Line2D([], [], color=SAMPLE_COLORS['0'], marker='+', linestyle='None', markersize=20, label='Non-Flooded Sample'),
        mlines.Line2D([], [], color=SAMPLE_COLORS['1'], marker='+', linestyle='None', markersize=20, label='Flooded Sample')
    ]
    
    stratum_legend_items = [
        mpatches.Patch(color=LABEL_CMAP(0), label='Non-Flooded Stratum'),
        mpatches.Patch(color=LABEL_CMAP(1), label='Flooded Stratum')
    ]
    
    water_legend_items = [
        mpatches.Patch(color=OSM_WATER_CMAP(1), label='OSM Water (masked)')
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
        # Create subplot with specified projection
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=plot_crs)
        
        # Get sample title from file path
        if isinstance(sample_path, str):
            title = Path(sample_path).stem
        else:
            title = sample_path.stem
            
        # Clean up the title: remove underscores and 'samples' word
        title = title.replace('_', ' ')
        title = title.replace('samples', '').replace('sample', '')
        title = title.strip()
        
        ax.set_title(title, fontsize=TITLE_SIZE)
        
        # Plot base layers (these are raster arrays with bounds in EPSG:4326)
        ax.imshow(label_band, cmap=LABEL_CMAP, alpha=0.6,
                extent=data_extent, transform=data_crs, origin='upper')
        
        ax.imshow(osm_water_band, cmap=OSM_WATER_CMAP, alpha=1, 
                extent=data_extent, transform=data_crs, origin='upper')
        
        # Read and plot sample points
        sample_counts = {0: 0, 1: 0}  # Count of samples by label
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
        
        # Only add lat/lon labels to the last subplot
        if i == len(sample_paths) - 1:
            gl = ax.gridlines(draw_labels=True, linewidth=0, linestyle='', alpha=0)
            gl.top_labels = False
            gl.left_labels = False
            gl.xformatter = LongitudeFormatter(number_format='.1f')
            gl.yformatter = LatitudeFormatter(number_format='.1f')
            gl.xlocator = mticker.LinearLocator(3) 
            gl.ylocator = mticker.LinearLocator(3)  
            
            # Increase font size for lat/lon labels
            gl.xlabel_style = {'size': AXIS_LABEL_SIZE}
            gl.ylabel_style = {'size': AXIS_LABEL_SIZE}
        else:
            # No gridlines for all subplots
            ax.gridlines(draw_labels=False, linewidth=0, linestyle='', alpha=0)
        
        # Create individual legend for this subplot showing sample counts and proportions
        total_samples = sample_counts[0] + sample_counts[1]
        
        if total_samples > 0:
            # Create simple text-only legend entries
            nf_legend = mlines.Line2D([], [], color='none', marker=None, linestyle='None', 
                                     label=f'NF: {sample_counts[0]}')
            
            f_legend = mlines.Line2D([], [], color='none', marker=None, linestyle='None', 
                                    label=f'F: {sample_counts[1]}')
            
            # Create a simple text-only legend at the bottom left
            subplot_legend = ax.legend(
                handles=[f_legend, nf_legend],
                loc='lower left',
                fontsize=SUBPLOT_LEGEND_TEXT_SIZE,
                frameon=False,  # No frame
                ncol=1,
                borderpad=0.2,
            )
            
            # Add the legend to the subplot
            ax.add_artist(subplot_legend)
    
    # Organize grid legends
    grid_legend_items = list(grid_legends.values())
    
    # Organize legend elements into columns
    legend_columns = [
        sample_legend_items,       # Column 1: Sample types
        stratum_legend_items,      # Column 2: Stratum types 
        water_legend_items,        # Column 3: Water  
        grid_legend_items          # Column 4: Grid types
    ]
    
    # Flatten legend elements for use in legend
    legend_elements = []
    for column in legend_columns:
        legend_elements.extend(column)
    
    # Add a single legend at the bottom of the figure with column organization
    legend = fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02), 
                  ncol=4,
                  fontsize=LEGEND_TEXT_SIZE,
                  columnspacing=1.0,
                  frameon=True, 
                  fancybox=True, 
                  framealpha=1.0,
                  edgecolor='black',
                  facecolor='white',
                  borderpad=0.8,
                  labelspacing=0.8,
                  handletextpad=0.5)
    
    # Set the legend frame linewidth
    legend.get_frame().set_linewidth(LEGEND_OUTLINE_WIDTH)
    
    if output_path:
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig 