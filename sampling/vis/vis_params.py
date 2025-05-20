from matplotlib.colors import ListedColormap

# Base Maps palettes
OSM_WATER_CMAP = ListedColormap(['none', '#A3A3A3'])  # Using 'none' for transparent pixels
WATER_COLOR = '#56b4e9'
LAND_COLOR = '#c99060'
LABEL_CMAP = ListedColormap([LAND_COLOR, WATER_COLOR])
SAMPLE_COLORS = {'0': '#009E73', '1': '#0173B2'}

# Text sizes for visualizations
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 16
LEGEND_TITLE_SIZE = 16
LEGEND_TEXT_SIZE = 16
SUBPLOT_LEGEND_TEXT_SIZE = 16

# Legend styling
LEGEND_OUTLINE_WIDTH = 0.8  

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
        "vmin": "auto",
        "vmax": "auto"
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
