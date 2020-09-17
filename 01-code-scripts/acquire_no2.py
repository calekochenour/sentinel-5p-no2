""" Downloads Sentinel-5P NO2 Level-2 processed data (netCDF files)

Run this file in the folder where the data should be downloaded to

Available Product Types:
Product                   Data file descriptor
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

Additional product information here:
    http://www.tropomi.eu/data-products/level-2-products

Note: Credentials for downloading data (lines 36, 75-76) have been replaced
      with '########'.

"""

# Imports
import os
import json
import csv


def execute_query():
    """Executes search query for files."""
    os.system(
        (
            f"wget --no-check-certificate --user=######## --password=######## "
            f"--output-document={query_results_file} "
            f'"https://s5phub.copernicus.eu/dhus/search?q=beginposition:[{time_span}] '
            f'AND (footprint:{escape_char}"Intersects({polygon}){escape_char}") '
            f'AND producttype:L2__{product_type}&rows=100&start=0&format=json"'
        )
    )

    return print("Completed query.")


def load_json(json_file):
    """Loads a JSON file."""
    # Load JSON
    with open(json_file) as file:
        loaded_json = json.load(file)

    return loaded_json


def write_to_csv():
    """Writes the file download links to a CSV file."""
    # Write UUID links to CSV file
    with open(
        f"query_links_{product_type}_{date_range}_{request_number}.csv",
        "w",
        newline="",
    ) as file:
        csv.writer(file).writerows(links_uuids)

    return print("Completed write to CSV.")


def download_files(link_list):
    """Downloads files based on links."""
    # Download each file from link
    for entry in link_list:
        os.system(
            (
                "wget --content-disposition --continue --user=######## "
                f"--password=######## {entry[0].get('href')}"
            )
        )

    return print("Completed download.")


def delete_files(file_extension):
    """Deletes files with a specified extention in the current directory."""
    # Get files to delete based on extension
    files_to_delete = [
        file
        for file in os.listdir(os.getcwd())
        if file.endswith(f".{file_extension}")
    ]

    # Delete files
    for file in files_to_delete:
        try:
            os.remove(file)
        except Exception as error:
            print(error)

    return print(f"Completed delete of .{file_extension} files.")


# Define Extent
# Centre County, PA
# lon_min = -78.4
# lon_max = -77.1
# lat_min = 40.6
# lat_max = 41.3

# South Korea
# lon_min = 125.0
# lon_max = 131.0
# lat_min = 33.1
# lat_max = 38.7

# North Korea
# lon_min = 124.1
# lon_max = 130.7
# lat_min = 37.6
# lat_max = 43.1

# North and South Korea
lon_min = 124.1
lon_max = 131.0
lat_min = 33.1
lat_max = 43.1

# Create bounding polygon
polygon = (
    "POLYGON(("
    f"{lon_min} {lat_min},"  # Bottom-Left
    f"{lon_min} {lat_max},"  # Top-Left
    f"{lon_max} {lat_max},"  # Top-Right
    f"{lon_max} {lat_min},"  # Bottom-Right
    f"{lon_min} {lat_min}))"  # Bottom-Left
)

# Wuhan bounding box
# polygon = (
#     "POLYGON(("
#     "113.70 30.11,"
#     "114.90 30.11,"
#     "114.90 30.97,"
#     "113.70 30.97,"
#     "113.70 30.11))"
# )

# Define time period for downloading data
# Data availability - 2018-06-28T10:24:07 - 2020-08-06T00:00:00
time_span = "2020-08-01T00:00:00.000Z TO 2020-08-03T00:00:00.000Z"

# Define all Sentinel-5P products
sentinel_products = {
    "UV Aerosol Index": "AER_AI",
    "Aerosol Layer Height": "AER_LH",
    "Carbon Monoxide": "CO____",
    "Cloud": "CLOUD_",
    "Formaldehyde": "HCHO__",
    "Methane": "CH4___",
    "Nitrogen Dioxide": "NO2___",
    "Sulphur Dioxide": "SO2___",
    "Ozone": "O3____",
    "Tropospheric Ozone": "O3_TCL",
}

# Get product for download
product_type = sentinel_products.get("Nitrogen Dioxide")

# Define initial query parameters
request_number = 0
num_processed_results = 0
start_date = time_span.split(" ")[0][0:10]
stop_date = time_span.split(" ")[2][0:10]
date_range = f"{start_date}_{stop_date}"
query_results_file = (
    f"query_results_{product_type}_{date_range}_{request_number}.json"
)
escape_char = "\\"

# Execute first query (create JSON file with query results)
execute_query()

# Open JSON
query_results = load_json(query_results_file)

# Get number of results from query (number of files to download)
num_total_results = int(query_results["feed"]["opensearch:totalResults"])

# Initialize list for file links and UUIDs
links_uuids = []

# Process JSON entries - run until all JSON entries have been processed
while num_processed_results < num_total_results:
    # Loop through each JSON entry
    for entry in range(0, len(query_results["feed"]["entry"])):
        # Check if the file contains NO2 data (from file title)
        if "NO2___" in query_results["feed"]["entry"][entry]["title"]:
            # Append file link and UUID to list and increment number processed
            links_uuids.append(
                [
                    query_results["feed"]["entry"][entry]["link"][0],
                    query_results["feed"]["entry"][entry]["link"][0][
                        "href"
                    ].split("'")[1],
                ]
            )
            num_processed_results += 1
    # Multiple pages of results - run if the number of query results
    #  exceeds the number processed
    if num_processed_results < num_total_results:
        # Increment request number, update JSON file name, execute new query
        request_number += 1
        query_results_file = (
            f"query_results_{product_type}_{date_range}_{request_number}.json"
        )
        execute_query()
        query_results = load_json(query_results_file)

# Write query results to CSV
write_to_csv()

# Download files from links
download_files(links_uuids)

# Delete CSV and JSON created during acquisition
for extension in ["json", "csv"]:
    delete_files(extension)
