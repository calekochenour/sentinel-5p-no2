""" Uploads NO2 Level-3 processed data to Box through the Box API

Note: The app.cfg file (line 25) must contain a valid Box developer token,
      client id, and client secret associated with a Box account for the
      script to work.

Note: Box folder IDs (lines 31 and 32) are commented out with '##########'.
      These must be replaced with valid folder IDs associated with a Box
      account for the script to work.
"""

# Imports
import os
import box

# Define local folders
local_folder_processed_netcdf = os.path.join(
    "03-processed-data", "netcdf", "south-korea"
)
local_folder_processed_geotiff = os.path.join(
    "03-processed-data", "raster", "south-korea"
)

# Create Box client session
client = box.create_session("app.cfg")

# Get root folder
root_folder = client.folder(folder_id="0").get()

# Define box folders
box_folder_processed_netcdf = client.folder(folder_id="############").get()
box_folder_processed_geotiff = client.folder(folder_id="############").get()

# Upload Level-3 netCDF and GeoTiff files To Box
box.upload_files_to_box(
    box_folder=box_folder_processed_netcdf,
    local_folder=local_folder_processed_netcdf,
    file_extension="nc",
)

box.upload_files_to_box(
    box_folder=box_folder_processed_geotiff,
    local_folder=local_folder_processed_geotiff,
    file_extension="tif",
)
