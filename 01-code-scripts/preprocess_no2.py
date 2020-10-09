""" Convert Sentinel-5P NO2 data from Level-2 to Level-3 processed """

# ENVIRONMENT SETUP
# Imports
import os
import glob
import sentinel as stl

# Get Level-2 netCDF files; retaining only those with data (file size > 0)
# netcdf_level2_folder = os.path.join("02-raw-data", "netCDF", "south-korea")
netcdf_level2_folder = os.path.join("02-raw-data", "netcdf", "singapore")
netcdf_level2_files = [
    file
    for file in glob.glob(os.path.join(netcdf_level2_folder, "*.nc"))
    if os.stat(file).st_size > 0
]

print("LEVEL-2 NETCDF FILES:")
for file in netcdf_level2_files:
    print(f"File: {file.split(os.sep)[-1]}")
    print(f"Date: {stl.extract_acquisition_time(file)}\n")

# PREPROCESS DATA
# Define harp operations parameters
level2_to_level3_parameters = {
    "quality_variable": "tropospheric_NO2_column_number_density_validity",
    "quality_comparison": ">",
    "quality_threshold": 75,
    # "bounding_box": (125.0, 33.1, 131.0, 38.7), # South Korea
    # Singapore with 10-cell buffer each direction
    "bounding_box": (103.6 - 0.25, 1.15 - 0.25, 104.1 + 0.25, 1.5 + 0.25),
    "cell_size": 0.025,
    "derive_variables": [
        # "tropospheric_NO2_column_number_density [molec/cm^2]",
        "latitude {latitude}",
        "longitude {longitude}",
    ],
    "keep_variables": [
        "latitude_bounds",
        "longitude_bounds",
        "latitude",
        "longitude",
        "tropospheric_NO2_column_number_density",
    ],
}

# Define output parameters
# level3_netcdf_output_folder = os.path.join(
#     "03-processed-data", "netcdf", "south-korea"
# )
level3_netcdf_output_folder = os.path.join(
    "03-processed-data", "netcdf", "singapore"
)
# level3_geotiff_output_folder = os.path.join(
#     "03-processed-data", "raster", "south-korea"
# )
level3_geotiff_output_folder = os.path.join(
    "03-processed-data", "raster", "singapore"
)
level3_output_type = "S5P-OFFL-L3-NO2"
level3_output_units = "mol-per-m2"  # alt. "molec-per-cm2"
level3_file_prefix = f"{level3_output_type}-{level3_output_units.upper()}"

# Convert level-2 to level-3 (creates netCDF and GeoTiff files)
try:
    stl.convert_level2_to_level3(
        level2_folder=netcdf_level2_files,
        import_operations=level2_to_level3_parameters,
        level3_netcdf_folder=level3_netcdf_output_folder,
        level3_geotiff_folder=level3_geotiff_output_folder,
        level3_prefix=level3_file_prefix,
    )
except Exception as error:
    print(error)
else:
    print("\nPreprocessing complete. Level-2 files converted to Level-3.")
