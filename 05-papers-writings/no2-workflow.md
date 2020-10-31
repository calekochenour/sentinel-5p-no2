# NO2 Data Workflow

This document specifies how to use the code in the `01-code-scripts/` folder to download and preprocess Sentienl-5P NO2 data.

## Download Data

Open `01-code-scripts/acquire_no2.py`.

Specify the bounding box with the following variables: `lon_min`, `lon_max`, `lat_min`, and `lat_max` (after line 100) for the study area of interest.

Navigate in the terminal to the folder where files will be downloaded to (e.g.):

```bash
$ cd ~/02-raw-data/netcdf/south-korea
```

Activate the Conda environment:

```bash
$ conda activate sentinel-5p
```    

Run the data acquisition script:

```bash
$ python ~/01-code-scripts/acquire_no2.py
```

## Convert Level-2 Processed Data to Level-3 Processed

Open `01-code-scripts/preprocess_no2.py`.

Set the `netcdf_level2_folder` variable to the path where the raw netCDF files are located (e.g. `os.path.join("02-raw-data", "netcdf", "south-korea")`).

Set the `bounding_box` key in the `level2_to_level3_parameters` variable to the bounding box of the study area (e.g. `"bounding_box": (125.0, 33.1, 131.0, 38.7)`).

Set the `level3_netcdf_output_folder` variable to the location where the Level-3 processed netCDF files will be output (e.g. `os.path.join("03-processed-data", "netcdf", "south-korea")`).

Set the `level3_geotiff_output_folder` variable to the location where the Level-3 processed GeoTiff files will be output (e.g. `os.path.join("03-processed-data", "raster", "south-korea")`).

Navigate in the terminal to the root directory of the repository (e.g.):

```bash
$ cd ~/sentienl-5p-no2
```

Run the preprocessing script:

```bash
$ python 01-code-scripts/preprocess_no2.py
```

## Calculate Statistics for Level-3 Data

Open `01-code-scripts/calculate_statistics.py`.

Set the `netcdf_level3_folder` variable to the location of the Level-3 netCDF files (e.g. `os.path.join("03-processed-data", "netcdf", "singapore")`).

Set the `dates` variable to the date range(s) for statistics (e.g. `[("2018-07-01", "2018-07-31"), ("2018-08-01", "2018-08-31")]`).

Set the `output_path` parameter in the `rd.export_array()` function to the location and file name for the output GeoTiff files (e.g. `os.path.join("03-processed-data", "raster", "south-korea", "statistics", f"{stat_name}-MOL-PER-M2.tif")`).

Navigate in the terminal to the root directory of the repository (e.g.):

```bash
$ cd ~/sentienl-5p-no2
```

Run the statistics script:

```bash
$ python 01-code-scripts/calculate_statistics.py
```
