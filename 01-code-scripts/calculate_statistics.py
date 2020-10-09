""" Calculates statistics of Sentinel-5P NO2 Level-3 processed data """

# ENVIRONMENT SETUP
# Imports
import os
import glob
import numpy as np
import radiance as rd
import sentinel as stl

# Define path to Level-3 netCDF files
# netcdf_level3_folder = os.path.join(
#     "03-processed-data", "netcdf", "south-korea"
# )
netcdf_level3_folder = os.path.join("03-processed-data", "netcdf", "singapore")

# PREPROCESS DATA
# Store Level-3 netCDF data in dictionary
no2_level3_dict = stl.store_level3_data(netcdf_level3_folder)

# PROCESS DATA
# Define start and end dates to extract
# start_date = "2019-01-01"
# end_date = "2019-01-31"

# Define list of start and end date tuples
dates = [
    ("2018-07-01", "2018-07-31"),
    ("2018-08-01", "2018-08-31"),
    ("2018-09-01", "2018-09-30"),
    ("2018-10-01", "2018-10-31"),
    ("2018-11-01", "2018-11-30"),
    ("2018-12-01", "2018-12-31"),
    ("2019-01-01", "2019-01-31"),
    ("2019-02-01", "2019-02-28"),
    ("2019-03-01", "2019-03-31"),
    ("2019-04-01", "2019-04-30"),
    ("2019-05-01", "2019-05-31"),
    ("2019-06-01", "2019-06-30"),
    ("2019-07-01", "2019-07-31"),
    ("2019-08-01", "2019-08-31"),
    ("2019-09-01", "2019-09-30"),
    ("2019-10-01", "2019-10-31"),
    ("2019-11-01", "2019-11-30"),
    ("2019-12-01", "2019-12-31"),
    ("2020-01-01", "2020-01-31"),
    ("2020-02-01", "2020-02-29"),
    ("2020-03-01", "2020-03-31"),
    ("2020-04-01", "2020-04-30"),
    ("2020-05-01", "2020-05-31"),
    ("2020-06-01", "2020-06-30"),
    ("2020-07-01", "2020-07-31"),
]

# Calculate statistics for all date ranges
for start_date, end_date in dates:
    # Set date range
    date_range = f"{start_date.replace('-', '')}-{end_date.replace('-', '')}"

    # Exract NO2 arrays to a list
    no2_array_list = stl.extract_arrays_to_list(
        no2=no2_level3_dict, dates=rd.create_date_list(start_date, end_date)
    )

    # Calulate mean, variance, and standard deviation
    statistics = {
        f"S5P-OFFL-L3-NO2-{date_range}-MEAN": rd.calculate_statistic(
            no2_array_list, statistic="mean"
        ),
        f"S5P-OFFL-L3-NO2-{date_range}-VARIANCE": rd.calculate_statistic(
            no2_array_list, statistic="variance"
        ),
        f"S5P-OFFL-L3-NO2-{date_range}-DEVIATION": rd.calculate_statistic(
            no2_array_list, statistic="deviation"
        ),
    }

    # EXPORT DATA
    # Create transform and metadata for export; assume the same for all files
    transform = stl.extract_no2_transform(
        glob.glob(os.path.join(netcdf_level3_folder, "*.nc"))[0]
    )

    metadata = rd.create_metadata(
        array=statistics.get(f"S5P-OFFL-L3-NO2-{date_range}-MEAN"),
        transform=transform,
        nodata=np.nan,
    )

    # Export arrays to GeoTiff
    for stat_name, stat_array in statistics.items():
        try:
            rd.export_array(
                array=stat_array,
                output_path=os.path.join(
                    "03-processed-data",
                    "raster",
                    # "south-korea",
                    "singapore",
                    "statistics",
                    f"{stat_name}-MOL-PER-M2.tif",
                ),
                metadata=metadata,
            )
        except Exception as error:
            print(error)

    # Ouput completion message
    print(f"Completed {date_range}\n")
