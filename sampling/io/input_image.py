from config import LABEL_BAND_NAME, WATER_BAND_NAME
import rasterio

def get_input_image(input_path: str):
    """
    Read the input GeoTIFF image and extract essential bands.
    
    Parameters:
    -----------
    input_path : str
        Path to the input GeoTIFF image
        
    Returns:
    --------
    tuple
        (image, metadata, label_band, osm_water_band)
        where metadata is a dict containing transform, crs, etc.
    """
    with rasterio.open(input_path) as src:
        # Read metadata
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'descriptions': src.descriptions
        }
        
        # Find LABEL and OSM_WATER bands by name
        label_idx = None
        water_idx = None
        
        for i, desc in enumerate(src.descriptions):
            if desc == 'LABEL':
                label_idx = i
                print(f"Found LABEL band at index {i+1}")
            elif desc == 'OSM_WATER':
                water_idx = i
                print(f"Found OSM_WATER band at index {i+1}")
        
        if label_idx is None or water_idx is None:
            raise ValueError("Error: LABEL and OSM_WATER bands not found in input image")
        
        # Read the bands
        label_band = src.read(label_idx + 1)
        osm_water_band = src.read(water_idx + 1)
        
        # Read all bands for the complete image
        image = src.read()
        
        print(f"Image shape: {image.shape}")
        print(f"LABEL band shape: {label_band.shape}")
        print(f"OSM_WATER band shape: {osm_water_band.shape}")
        
        return image, metadata, label_band, osm_water_band