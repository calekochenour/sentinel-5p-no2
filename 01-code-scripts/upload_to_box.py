"""
#!/usr/bin/env python
# coding: utf-8
"""
# ENVIRONMENT SETUP
# Imports
import os
import earthpy as et
import box

# Set working directory
working_directory = os.path.join(
    et.io.HOME,
    "PSU",
    "08-covid19-remote-sensing-fusion",
    "00-git-repos",
    "sentinel-5p-demo",
)

os.chdir(working_directory)

# Define path to data for upload
local_folder_raw_data = os.path.join("south-korea", "02-raw-data")
local_folder_processed_data = os.path.join("south-korea", "03-processed-data")

# Create client session
client = box.create_session("app.cfg")

# Display user properties
# for user_attribute in user:
#     print(user_attribute)

# Get root folder
root_folder = client.folder(folder_id="0").get()
# root_folder

# Display folder properties
# for folder_attribute in root_folder:
#     print(folder_attribute)

# Display all folders in root
# box.display_all_folders(client_session=client, root_folder=0)

# # Show specific folder, with some hierarchy
# box.display_specific_folder(client_session=client, root_folder=0,
#                             target_folder_name='centre-county-no2')

# # Show specific folder, with some hierarchy
# box.display_specific_folder(client_session=client, root_folder=0, target_folder_name='geotiff')
#
# # Show specific folder, with some hierarchy
# box.display_specific_folder(client_session=client, root_folder=0, target_folder_name='02-processed-data')
#
# # Show specific folder, with some hierarchy
# box.display_specific_folder(client_session=client, root_folder=0, target_folder_name='sentinel-5p-no2')
#
# # Show pennsylvania-centre-county, with some hierarchy
# box.display_specific_folder(client_session=client, root_folder=0, target_folder_name='pennsylvania-centre-county')
#
# Get south-korea folder (for upload) based on folder ID
# Raw data
box_folder_raw_data = client.folder(folder_id="121409425698").get()

# Processed data / netcdf
box_folder_processed_netcdf = client.folder(folder_id="121409975164").get()

# Processed data / geotiff
box_folder_processed_geotiff = client.folder(folder_id="121410607247").get()

# Upload Level-2 netCDF files to BOX
# box.upload_files_to_box(
#     box_folder=box_folder_raw_data,
#     local_folder=local_folder_raw_data,
#     file_extension='nc'
# )

# Upload Level-3 netCDF files To BOX
box.upload_files_to_box(
    box_folder=box_folder_processed_netcdf,
    local_folder=local_folder_processed_data,
    file_extension="nc",
)

# Upload Level-3 GeoTiff files To BOX
box.upload_files_to_box(
    box_folder=box_folder_processed_geotiff,
    local_folder=local_folder_processed_data,
    file_extension="tif",
)
