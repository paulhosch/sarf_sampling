from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker

def add_grid_lines(ax):
    """
    Add gridlines with formatted latitude and longitude labels to a cartopy map.
    
    Parameters:
    -----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        The cartopy axes to add gridlines to
    """
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter(number_format='.1f')
    gl.yformatter = LatitudeFormatter(number_format='.1f')
    gl.xlocator = mticker.LinearLocator(4) 

def get_square_extent(bounds):
    """
    Create a square extent centered on the input bounds.
    
    This ensures that the map has equal dimensions in both directions,
    using the larger of the width or height as the common dimension.
    
    Parameters:
    -----------
    bounds : tuple
        (left, bottom, right, top) boundaries in map units
        
    Returns:
    --------
    list
        [left, right, bottom, top] extent for cartopy's set_extent method
    """
    # Create square bounds by finding the larger dimension
    # bounds = (left, bottom, right, top)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    
    # Use the larger dimension to create a square
    max_dimension = max(width, height)
    square_bounds = [
        center_x - max_dimension / 2,  # left
        center_y - max_dimension / 2,  # bottom
        center_x + max_dimension / 2,  # right
        center_y + max_dimension / 2   # top
    ]
    square_extent = [square_bounds[0], square_bounds[2], square_bounds[1], square_bounds[3]]  # Square extent
    return square_extent

def get_data_extent(bounds):
    """
    Convert rasterio bounds to cartopy extent format.
    
    Parameters:
    -----------
    bounds : tuple
        (left, bottom, right, top) boundaries in map units
        
    Returns:
    --------
    list
        [left, right, bottom, top] extent for cartopy's set_extent method
    """
    data_extent = [bounds[0], bounds[2], bounds[1], bounds[3]]
    return data_extent

