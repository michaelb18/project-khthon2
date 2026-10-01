import openeo
import tempfile
import tifffile
import numpy as np
import os
import rasterio
from datetime import datetime
import math
# First, we connect to the back-end and authenticate. 
connection = openeo.connect("openeofed.dataspace.copernicus.eu")
connection.authenticate_oidc()

from pyproj import Transformer
import pyproj
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import mlstac
import torch
import sen2sr

def meters2latlon(center_lat, center_lon, height, width):
    lat_degree_m = 111111
    lon_degree_m = 111111 * math.cos(math.radians(center_lat))

    lat_margin = (height / 2) / lat_degree_m
    lon_margin = (width / 2) / lon_degree_m

    lon_min = center_lon - lon_margin
    lon_max = center_lon + lon_margin
    lat_min = center_lat - lat_margin
    lat_max = center_lat + lat_margin

    return (lon_min, lat_min, lon_max, lat_max)

def ll2px(rds, lat_list, lon_list):
    """
    Converts lists of latitudes and longitudes to pixel row and column indices.
    
    Parameters:
        rds (xarray.DataArray): The rioxarray-opened raster object.
        lat_list (list): List of latitude floats.
        lon_list (list): List of longitude floats.
        
    Returns:
        tuple: (rows, cols) as lists of integer pixel indices.
    """
    # Get spatial metadata
    dst_crs = rds.rio.crs
    transform = rds.rio.transform()
    inv_transform = ~transform

    # Set up coordinate transformer
    transformer = pyproj.Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    
    rows = []
    cols = []
    
    # Process coordinate pairs
    for lat, lon in zip(lat_list, lon_list):
        x, y = transformer.transform(lon, lat)
        col, row = inv_transform * (x, y)
        
        rows.append(int(row))
        cols.append(int(col))
        
    return rows, cols

def get_image(center_lat, center_lon, height, width, save_path = 'champs3.tiff', days = 30, time = None):
    
    if time is None:
        time = datetime.now().date()
    lon_min, lat_min, lon_max, lat_max = meters2latlon(center_lat, center_lon, height, width)

    start = time-timedelta(days=(days))
    end = time+timedelta(days=(days))
    if end > datetime.now().date():
        end = datetime.now().date()
        start = datetime.now().date() - timedelta(days=(days))
        
    temporal_extent = [str(start),str(end)]
    spatial_extent = {
        "west": lon_min, "south": lat_min,
        "east": lon_max, "north": lat_max}
    sentinel2_cube = connection.load_collection(
        "SENTINEL2_L2A", #collection chosen
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
        max_cloud_cover=50
    )

    #sentinel2_cube = sentinel2_cube.process("mask_scl_dilation", data=sentinel2_cube, scl_band_name="SCL")
    median_image = sentinel2_cube.reduce_dimension(dimension="t", reducer="median")

    from openeo.processes import ProcessBuilder

    # define child process, use ProcessBuilder
    def scale_function(x: ProcessBuilder):
        return x * 0.0001

    # Convert to reflectance (simple multiplication)
    #print("Converting to reflectance...")
    #reflectance_cube= median_image.apply(scale_function)

    #def scale_function(x: ProcessBuilder):
    #    return x.linear_scale_range(0, 1, 0, 255)

    # apply scale_function to all pixels
    #visual_image= reflectance_cube.apply(scale_function)
    final_result = median_image.save_result(format="GTiff")
 
    job_title = "Field_Observation"

    job = final_result.create_job(
                    title=job_title,
                    description="This pipeline downloads and processes images of specific fields in France"
    )

    def progress_callback(status):
        # This gets called whenever the status is polled
        # We can fetch more detail directly from the job object if needed
        info = job.describe()
        progress = info.get('progress', 'Unknown %')
        print(f"🛰️ Status: {status} | Progress: {progress} | Job ID: {job.job_id}")


    try:    
        result = job.start_and_wait(
                        print=progress_callback,
                        max_poll_interval=30, 
                        connection_retry_interval=60  
                    )
        print(f"✅ Job started with ID: {job.job_id}")
    except Exception as e:
        print(f"the job failed {e}")


    with tempfile.TemporaryDirectory() as download_dir:
        download_dir = os.path.join(download_dir, save_path)
        image = job.download_results(download_dir)
        import rioxarray

    #    # Open the image with rioxarray
        img = rioxarray.open_rasterio(download_dir)

    #with tempfile.TemporaryDirectory() as download_dir:
    #    download_dir = os.path.join(download_dir, 'champs3.tiff')
    #    image = final_result.download(download_dir)
    #    import rioxarray

    #    img = rioxarray.open_rasterio(download_dir)

    
    return img

def label_sentinel2_with_lucas(
    img,
    class_columns: list[str] | None = None,
) -> dict[str, np.ndarray]:
    from read_lucas import read_lucas
    from shapely.geometry import box
    from shapely.ops import transform as shapely_transform
    from rasterio.features import rasterize
    from pyproj import Transformer

    if class_columns is None:
        class_columns = ["LC1", "LC1_SPEC", "LC2"]

    lucas_gdf = read_lucas()

    img_crs = img.rio.crs
    minx, miny, maxx, maxy = img.rio.bounds()
    to_wgs84 = Transformer.from_crs(img_crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = to_wgs84.transform(minx, miny)
    lon_max, lat_max = to_wgs84.transform(maxx, maxy)
    bbox = box(lon_min, lat_min, lon_max, lat_max)

    subset = lucas_gdf[lucas_gdf.geometry.intersects(bbox)].copy()

    nrows, ncols = img.shape[1], img.shape[2]
    transform = img.rio.transform()
    to_img_crs = Transformer.from_crs("EPSG:4326", img_crs, always_xy=True)

    masks = {}
    for col in class_columns:
        if col not in subset.columns:
            continue

        valid = subset[col].notna()
        if valid.sum() == 0:
            masks[col] = np.full((nrows, ncols), None, dtype=object)
            continue

        unique_vals = subset.loc[valid, col].unique()
        val_to_int = {v: i + 1 for i, v in enumerate(unique_vals)}

        shapes = []
        for _, row in subset[valid].iterrows():
            geom_img = shapely_transform(
                lambda x, y: to_img_crs.transform(x, y), row.geometry
            )
            shapes.append((geom_img, val_to_int[row[col]]))

        int_mask = rasterize(
            shapes,
            out_shape=(nrows, ncols),
            transform=transform,
            fill=0,
            dtype=np.int32,
        )

        str_mask = np.full((nrows, ncols), None, dtype=object)
        for val, code in val_to_int.items():
            str_mask[int_mask == code] = val

        masks[col] = str_mask

    return masks


def turn_into_image(image2):
    r = np.array(image2.sel(band=4)).squeeze()   # B04 — Red
    g = np.array(image2.sel(band=3)).squeeze()   # B03 — Green
    b = np.array(image2.sel(band=2)).squeeze()   # B02 — Blue
    img = np.stack([r, g, b], axis=-1).astype(np.float32)

    min_val, max_val = np.percentile(img, (2, 98))
    rgb_image = np.clip((img - min_val) / (max_val - min_val), 0, 1)

    return rgb_image

def download_sen2sr(output_dir = 'model/SEN2SRLite_RGBN'):
    mlstac.download(file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json", output_dir=output_dir)
    
def to_2_5_m(image, input_dir = "model/SENSRLite_RGBN"):
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    r = np.array(image.sel(band=4)).squeeze()   # B04 — Red
    g = np.array(image.sel(band=3)).squeeze()   # B03 — Green
    b = np.array(image.sel(band=2)).squeeze()   # B02 — Blue
    b8 = np.array(image.sel(band=8)).squeeze()   # B02 — Blue
    img = torch.from_numpy(np.stack([r, g, b, b8], axis=0).astype(np.float32))

    dummy_s2_data = torch.nn.functional.pad(img, (0, (1 << (img.shape[-1] - 1).bit_length()) - img.shape[-1], 0, (1 << (img.shape[-2] - 1).bit_length()) - img.shape[-2]), mode='constant')
    
    print(dummy_s2_data.shape)
    # 3. Convert to PyTorch Tensor
    X = dummy_s2_data.float().to(device)

    # 4. Load the model via the reader and run inference
    model = mlstac.load("model/SEN2SRLite_RGBN").compiled_model(device=device)
    with torch.no_grad():
        super_resolved_tensor = sen2sr.predict_large(model=model, X=X, overlap=32)
        #super_resolved_tensor = model(X[None]).squeeze(0) # Returns shape (4, 512, 512)
        
    img = super_resolved_tensor[:3, :img.shape[1] * 4, :img.shape[2] * 4].permute(1, 2, 0).numpy()
    min_val, max_val = np.percentile(img, (2, 98))
    rgb_image = np.clip((img - min_val) / (max_val - min_val), 0, 1)
    
    return rgb_image