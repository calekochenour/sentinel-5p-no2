"""
-------------------------------------------------------------------------------
 Converts Sentinel-5P data from Level-2 procssed to Level-3 processed.

 Sentinel-5P Level-2 processed data exists in non-square pixels. Converting
 from Level-2 to Level-3 involves re-gridding/re-binning the data to square
 pixels.

 Sentinel-5P Product Types:
 Product                   Data File Descriptor
 UV Aerosol Index          AER_AI
 Aerosol Layer Height      AER_LH
 Carbon Monoxide (CO)      CO____
 Cloud                     CLOUD_
 Formaldehyde (HCHO)       HCHO__
 Methane (CH4)             CH4___
 Nitrogen Dioxide (NO2)    NO2___
 Sulphur Dioxide (SO2)     SO2___
 Ozone (O3)                O3____
 Tropospheric Ozone        O3_TCL

 This script uses variables from the sentinel.py module that store input and
 output folder paths, as well as file naming output options. In-script
 configurations include the 'level2_to_level3_paramater_options' variable,
 which stores the harp 'bin_spatial()' operations for converting Level-2 to
 Level-3. Setting the 'country_name', 'product_name', and 'unit' variables
 will allow the script to preprocess the data in the desired way.

 Available country names:
   - 'Singapore'
   - 'South Korea'

 Available product types:
   - UV Aerosol Index
   - Aerosol Layer Height
   - Carbon Monoxide (CO)
   - Cloud
   - Formaldehyde (HCHO)
   - Methane (CH4)
   - Nitrogen Dioxide (NO2)
   - Sulphur Dioxide (SO2)
   - Ozone (O3)
   - Tropospheric Ozone

 Available units:
   - 'Mole'
   - 'Molecule'
-------------------------------------------------------------------------------
"""
# -------------------------ENVIRONMENT SETUP--------------------------------- #
# Import packages
import os
import glob
import sentinel as stl

# -------------------------USER-DEFINED VARIABLES---------------------------- #
# Set country name
# country_name = "Singapore"
country_name = "South Korea"

# Set product name
product_name = "Carbon Dioxide"
# product_name = "Nitrogen Dioxide"

# Set units for output file name
unit = "Mole"
# units = "Molecule"

# Define Level-2 to Level-3 re-gridding parameter options
level2_to_level3_parameter_options = {
    "UV Aerosol Index": {},
    "Aerosol Layer Height": {},
    "Carbon Monoxide": {
        "quality_variable": "CO_column_number_density_validity",
        "quality_comparison": ">=",
        "quality_threshold": 50,
        "bounding_box": stl.BOUNDING_BOXES.get(country_name),
        "cell_size": 0.025,
        "derive_variables": ["latitude {latitude}", "longitude {longitude}"],
        "keep_variables": [
            "latitude_bounds",
            "longitude_bounds",
            "latitude",
            "longitude",
            "CO_column_number_density",
        ],
    },
    "Cloud": {},
    "Formaldehyde": {},
    "Methane": {},
    "Nitrogen Dioxide": {
        "quality_variable": "tropospheric_NO2_column_number_density_validity",
        "quality_comparison": ">",
        "quality_threshold": 75,
        "bounding_box": stl.BOUNDING_BOXES.get(country_name),
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
    },
    "Sulphur Dioxide": {},
    "Ozone": {},
    "Tropospheric Ozone": {},
}

# -------------------------DATA PREPROCESSING-------------------------------- #
# Get Level-2 netCDF files; retaining only those with data (file size > 0)
netcdf_level2_files = [
    file
    for file in glob.glob(
        os.path.join(
            stl.NETCDF_LEVEL2_FOLDERS.get(country_name).get(product_name),
            "*.nc",
        )
    )
    if os.stat(file).st_size > 0
]

print("LEVEL-2 NETCDF FILES:")
for file in netcdf_level2_files:
    print(f"File: {file.split(os.sep)[-1]}")
    print(f"Date: {stl.extract_acquisition_time(file)}\n")

# Set Level-2 to Level-3 re-gridding parameters
level2_to_level3_parameters = level2_to_level3_parameter_options.get(
    product_name
)

# Set Level-3 output folders (netCDF and GeoTiff)
level3_netcdf_output_folder = stl.LEVEL3_NETCDF_OUTPUT_FOLDERS.get(
    country_name
).get(product_name)
level3_geotiff_output_folder = stl.LEVEL3_GEOTIFF_OUTPUT_FOLDERS.get(
    country_name
).get(product_name)

# Set output file name prefix
level3_output_type = stl.LEVEL3_OUTPUT_TYPES.get(product_name)
level3_output_units = stl.LEVEL3_OUTPUT_UNITS_OPTIONS.get(unit)
level3_file_prefix = f"{level3_output_type}-{level3_output_units.upper()}"

# Convert level-2 to level-3 (creates netCDF and GeoTiff files)
try:
    if product_name == "Carbon Dioxode":
        stl.convert_level2_to_level3_co(
            level2_folder=netcdf_level2_files,
            import_operations=level2_to_level3_parameters,
            level3_netcdf_folder=level3_netcdf_output_folder,
            level3_geotiff_folder=level3_geotiff_output_folder,
            level3_prefix=level3_file_prefix,
        )
    elif product_name == "Nitrogen Dioxide":
        stl.convert_level2_to_level3(
            level2_folder=netcdf_level2_files,
            import_operations=level2_to_level3_parameters,
            level3_netcdf_folder=level3_netcdf_output_folder,
            level3_geotiff_folder=level3_geotiff_output_folder,
            level3_prefix=level3_file_prefix,
        )
    else:
        raise ValueError(
            (
                "Invalid product name. Must be Carbon Monoxide or Nitrogen "
                "Dioxide."
            )
        )
except Exception as error:
    print(error)
else:
    print("\nPreprocessing complete. Level-2 files converted to Level-3.")

# -------------------------SCRIPT COMPLETION--------------------------------- #
print("\n")
print("-" * (18 + len(os.path.basename(__file__))))
print(f"Completed script: {os.path.basename(__file__)}")
print("-" * (18 + len(os.path.basename(__file__))))
