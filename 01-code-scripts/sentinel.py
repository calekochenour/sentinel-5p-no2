""" Module to work with Sentinel-5p data """

import os
import re
import glob
from datetime import timedelta
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import numpy.ma as ma
from scipy.interpolate import UnivariateSpline
import pandas as pd
from pandas.io.json import json_normalize
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.transform import from_origin
import rasterstats as rs
import harp
from harp._harppy import NoDataError
import earthpy.plot as ep
import radiance as rd


def create_resample_operations(
    bounding_box, cell_size=0.025, quality_value=50
):
    """Returns a string that is formatted to
    work with the harp.import_product() function.

    Parameters
    ----------
    bounding_box : tuple (of int or float)
        Tuple (min longitude, min latitude,
        max longitude, max latitude) or
        (left, bottom, right, top) for
        the study area.

    cell_size : int or float
        Cell size in degrees (at the equator) of
        the output/resampled data. Default value
        is 0.025 (similar to native data resolution).

    quality_value : int
        Value used to filter data on. Only data
        with QA values greater that the specified
        value will be included. Valid values are
        0-100. Default value is 50.

    Returns
    -------
    operations : str
        String formatted to work with the
        harp.import_product() function.

    Example
    -------
        >>> # Create operations
        >>> no2_operations = no2_resample_operations(
        ...     bounding_box=(113.5, 30, 115.5, 31),
        ...     cell_size=0.025,
        ...     quality_value=50
        ... )
        >>> # Display operations
        >>> no2_operations
        'tropospheric_NO2_column_number_density_validity > 50;
         latitude > 30 [degree_north];
         latitude < 31 [degree_north];
         longitude > 113.5 [degree_east];
         longitude < 115.5 [degree_east];
         bin_spatial(41, 30, 0.025, 81, 113.5, 0.025);
         derive(latitude {latitude});
         derive(longitude {longitude})'
    """
    # Compute number of lat/lon grid cells in each column/row
    num_latitude_cells = int(
        (bounding_box[3] - bounding_box[1]) / cell_size + 1
    )
    num_longitude_cells = int(
        (bounding_box[2] - bounding_box[0]) / cell_size + 1
    )

    # Define operations
    operations = (
        # Filter cells for data quality
        f"tropospheric_NO2_column_number_density_validity > {quality_value}; "
        # Filter cells by lat/lon bounds
        f"latitude > {bounding_box[1]} [degree_north]; "
        f"latitude < {bounding_box[3]} [degree_north]; "
        f"longitude > {bounding_box[0]} [degree_east]; "
        f"longitude < {bounding_box[2]} [degree_east]; "
        # Re-sample grid
        f"bin_spatial({num_latitude_cells}, {bounding_box[1]}, {cell_size}, "
        f"{num_longitude_cells}, {bounding_box[0]}, {cell_size}); "
        # Derive pixel centers for re-sampled grid
        "derive(latitude {latitude}); "
        "derive(longitude {longitude})"
    )

    # Return operations
    return operations


def remove_trailing(operations_string):
    """Removes unnecessary characters from an import operations string."""
    # Remove characters
    if operations_string[-1] == ";":
        operations = operations_string[:-1]
    elif operations_string[-2:] == "; ":
        operations = operations_string[:-2]
    else:
        pass

    # Return updated operations string
    return operations


def create_import_operations(
    quality_variable=None,
    quality_comparison=None,
    quality_threshold=None,
    bounding_box=None,
    cell_size=None,
    derive_variables=None,
    keep_variables=None,
):
    """Returns a string that is formatted to
    work with the harp.import_product() function
    operations parameter.

    Documentation:
        https://stcorp.github.io/harp/doc/html/operations.html

    Parameters
    ----------
    quality_variable : str
        Variable name of the data quality indicator of interest.
        Data will be filtered based on this variable.

    quality_comparison : str
        Operator used to compare/filter data based on the quality_variable
        and quality_threshold. Value values are:

        '==' (equal to, data == 5)
        '!=' (not equal to, data != 5)
        '<'  (less than, data < 5)
        '<=' (less than or equal to, data <= 5)
        '>=' (greater than or equal to, data >= 5)
        '>'  (greater than, data > 5)
        '=&' (bitfield, data =& 5, True if both bits 1 and 3 in data are set)
        '!&' (bitfield, data !& 5, True if neither bits 1 or 3 in data are set)

    quality_threshold : int or float
        Threshold value to filter the data on.

    bounding_box : tuple of int or float
        Bounding box to filter the data on, formatted as
        (longitude_min, latitude_min, longitude_max, latitude_max).

    cell_size : int or float
        Cell size in degrees for the resampled grid.

    derive_variables : list of str
        List of variables to derive during import, with units and/or
        dimensions as applicable.

    keep_variables : list of str
        List of variables to keep during import.

    Returns
    -------
    operations : str
        String of the harp import operations.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Define quality component
    if quality_variable and quality_comparison and quality_threshold:
        quality_str = (
            f"{quality_variable} {quality_comparison} {quality_threshold}; "
        )
    else:
        quality_str = None

    # Define geographic bounds and bin_spatial() components
    if bounding_box and cell_size:
        lat_min_str = f"latitude > {bounding_box[1]} [degree_north]; "
        lat_max_str = f"latitude < {bounding_box[3]} [degree_north]; "
        lon_min_str = f"longitude > {bounding_box[0]} [degree_east]; "
        lon_max_str = f"longitude < {bounding_box[2]} [degree_east]; "
        geographic_bounds_str = (
            lat_min_str + lat_max_str + lon_min_str + lon_max_str
        )

        # Compute number of lat/lon grid cells in each column/row
        num_latitude_cells = int(
            (bounding_box[3] - bounding_box[1]) / cell_size + 1
        )
        num_longitude_cells = int(
            (bounding_box[2] - bounding_box[0]) / cell_size + 1
        )

        # Deifine parameters for bin_spatial() function
        bin_spatial_params = map(
            str,
            [
                num_latitude_cells,
                bounding_box[1],
                cell_size,
                num_longitude_cells,
                bounding_box[0],
                cell_size,
            ],
        )

        # Define bin_spatial() string
        bin_spatial_str = f"bin_spatial({', '.join(bin_spatial_params)}); "

    else:
        geographic_bounds_str = None
        bin_spatial_str = None

    # Define derive variables component
    if derive_variables:
        derive_str = "".join(
            [f"derive({variable}); " for variable in derive_variables]
        )
    else:
        derive_str = None

    # Define keep variables component
    if keep_variables:
        keep_str = f"keep({', '.join(keep_variables)});"
    else:
        keep_str = None

    # Create list of operation components
    operations_list = [
        quality_str,
        geographic_bounds_str,
        bin_spatial_str,
        derive_str,
        keep_str,
    ]

    # Join operations into single string
    operations_string = "".join(
        [operation for operation in operations_list if operation]
    )

    # Remove unnecessary trailing characters
    operations = (
        remove_trailing(operations_string)
        if operations_string
        else operations_string
    )

    # Return operations
    return operations


def extract_acquisition_time(netcdf4_path):
    """Extracts and returns the acqusition date
    and time for a Sentinel-5P netCDF4 file.

    Parameters
    ----------
    netcdf4_path : str
        Path to the netCDF4 file.

    Returns
    -------
    acquisiton_time : str
        Acqusition start time for the data,
        formatted as YYYY-MM-DDTHHMMSS'

    Example
    -------
        >>> # Extract time
        >>> extract_acquisition_time(nc_file_path)
        '2019-02-02-T052350Z'
    """
    # Extract acquisition start time
    with Dataset(netcdf4_path, "r") as netcdf4:
        acquisition_time = netcdf4.groups["PRODUCT"]["time_utc"][0][0][
            :19
        ].replace(":", "")

    # Separate y/m/d from time and add indicator of Zulu time
    acquisition_time = f"{acquisition_time[:10]}-{acquisition_time[10:]}Z"

    # Return start time
    return acquisition_time


def resample_netcdf4(
    netcdf4_path,
    resample_operations,
    export_folder,
    file_prefix,
    acquisition_time,
):
    """Resamples a Sentinel-5P netCDF4 file to a
    spatially-uniform grid cell size and exports
    the resampled file to netCDF4 format.

    Parameters
    ----------
    netcdf4_path : str
        Path to the netCDF4 file.

    resample_operations : str
        Operations used to filter and resample
        the netCDF4 file.

    export_folder : str
        Folder (exluding filename) for the
        exported netCDF4 file.

    acquisition_time : str
        Acquisition time of the data, for use
        in output file naming.

    Returns
    -------
    out_message : str
        Path of exported file if successful, otherwise
        message indicating failure to export the file.

    Example
    -------
        >>> # Resample netCDF file
        >>> resampled_path = resample_netcdf4(
        ...     nc_file, no2_operations,
        ...     export_folder="02-processed-data",
        ...     acquisition_time="2020-05-05-T052030Z"
        ... )
        "02-processed-data/S5P_OFFL_L3_NO2_2020-05-05-T052030Z.nc"
    """
    # Import netCDF file into harp, with operations
    resampled_data = harp.import_product(netcdf4_path, resample_operations)

    # Define export path
    export_path = os.path.join(
        export_folder, f"{file_prefix}-{acquisition_time}.nc"
    )

    try:
        # Export re-sampled harp product to netCDF
        harp.export_product(resampled_data, export_path, file_format="netcdf")

    except Exception:
        # Define failure message
        out_message = "Failed to export."

    else:
        # Define success message
        out_message = export_path
        print(f"Exported: {os.path.split(export_path)[-1]}")

    # Return message
    return out_message


def extract_no2_data(netcdf_path):
    """Extracts NO2 data from a Sentinel-5P netCDF4 file
    to a NumPy array and fills no data with NaN values.

    Intended for use with resampled data (Sentinel-5P L3).

    Parameters
    ----------
    netcdf_path : str
        Path to the netCDF file.

    Returns
    -------
    no2_data_filled : numpy array
        Array containing extracted NO2 values and no
        data values filled with NaN, if applicable.

    Example
    -------
        >>> # Define file path
        >>> netcdf_file = os.path.join(
        ...     "02-processed-data",
        ...     "S5P_OFFL_L3_NO2_2020-05-05-T052030Z.nc"
        ... )
        >>> # Extract NO2 data
        >>> no2_data = extract_no2_data(netcdf_file)
    """
    # Open .nc file as netCDF Dataset
    with Dataset(netcdf_path, "r") as netcdf_resampled:

        # Get NO2 data
        no2_data = netcdf_resampled.variables.get(
            "tropospheric_NO2_column_number_density"
        )[0]

        # Change fill value to NaN and fill array
        if isinstance(no2_data, np.ma.core.MaskedArray):

            # Change fill value to NaN
            ma.set_fill_value(no2_data, np.nan)

            # Fill masked values with NaN
            no2_data_filled = no2_data.filled()

        else:
            no2_data_filled = np.copy(no2_data)

        # Flip array to correct orientation
        no2_data_filled_flipped = np.flipud(no2_data_filled)

    # Return array
    return no2_data_filled_flipped


def store_no2_data(no2_data, no2_dict, acquisition_time):
    """Stores NO2 array in a dictionary, indexed by
    data acquisition year, month, day, and time.

    Parameters
    ----------
    no2_data : numpy array
        Array containing NO2 values.

    no2_dict : dict
        Dictionary for storing extracted NO2 data.

    acquisition_time : str
        Acquisition time of the data, used to index
        the extracted data in the dictionary.

    Returns
    -------
    message : str
        Message indicating success or failure to
        add NO2 data to the dictionary.

    Example
    -------
        >>> # Define file path
        >>> netcdf_file = os.path.join(
        ...     "02-processed-data",
        ...     "S5P_OFFL_L3_NO2_2020-05-05-T052030Z.nc"
        ... )
        >>> # Extract NO2 data
        >>> no2_data = extract_no2_data(netcdf_file)
        >>> # Initialize NO2 dictionary
        >>> no2 = {}
        >>> # Store NO2 data
        >>> store_no2_data(
        ...     no2_data, no2, "2020-05-05-T052030Z"
        ... )
        Added 2020-05-05-T052030Z array to dictionary.
    """
    # Split acquisition time into components
    filename_split = os.path.splitext(acquisition_time)[0].split("-")
    year = filename_split[-4]
    month = filename_split[-3]
    day = filename_split[-2]
    time = filename_split[-1]

    # Add year to dictionary if not existing key
    if year not in no2_dict.keys():
        no2_dict[year] = {}

    # Add month to dictionary if not existing key
    if month not in no2_dict.get(year).keys():
        no2_dict[year][month] = {}

    # Add day to dictionary if not existing key
    if day not in no2_dict.get(year).get(month).keys():
        no2_dict[year][month][day] = {}

    # Store NO2 array, indexed by acquistion year, month, day, and time
    if time not in no2_dict.get(year).get(month).get(day).keys():
        no2_dict[year][month][day][time] = no2_data

        # Define output message
        message = f"Added {acquisition_time} array to dictionary."

    else:
        # Define output message
        message = (
            f"{acquisition_time} index already in dictionary. Skipping..."
        )

    # Return message
    return message


def extract_no2_transform(netcdf_path):
    """Extracts and returns the transform from
    a netCDF file.

    Intended for use with resampled data (Sentinel-5P L3).

    Parameters
    ----------
    netcdf_path : str
         Path to the netCDF file.

    Returns
    -------
    transform : affine.Affine object
        Geographic transform for the file.

    Example
    ------
        >>> # Define file path
        >>> netcdf_file = os.path.join(
        ...     "02-processed-data",
        ...     "S5P_OFFL_L3_NO2_2020-05-05-T052030Z.nc"
        ... )
        >>> # Create transform
        >>> transform = extract_no2_transform(netcdf_file)
        >>> # Display transform
        >>> transform
        Affine(0.025, 0.0, -73.0,
               0.0, -0.025, 42.5)
    """
    # Open .nc file as netCDF Dataset
    with Dataset(netcdf_path, "r") as netcdf_resampled:

        # Get longitude pixel bounds
        # Lower-left, lower-right, upper-right, upper-left
        longitude_bounds = netcdf_resampled.variables.get("longitude_bounds")[
            :
        ]

        # Get longitude pixel bounds
        # Lower-left, lower-right, upper-right, upper-left
        latitude_bounds = netcdf_resampled.variables.get("latitude_bounds")[:]

        # Get top-left corner of image
        longitude_min = longitude_bounds.min()
        latitude_max = latitude_bounds.max()

        # Define cell spacing (degrees)
        column_spacing = round(
            longitude_bounds[0][-1] - longitude_bounds[0][0], 6
        )
        row_spacing = round(latitude_bounds[0][-1] - latitude_bounds[0][0], 6)

        # Define transform
        # Top-left corner: west, north, and pixel size: xsize, ysize
        transform = from_origin(
            longitude_min, latitude_max, column_spacing, row_spacing
        )

    # Return transform
    return transform


def extract_acquisition_time_processed(file_path):
    """Extracts the acquistion time for a Level-3
    (L3) processed Sentinel-5P NO2 data file.

    Parameters
    ----------
    file_path : str
        Path to the L3 netCDF file.

    Returns
    -------
    acquisition_time : str
        Acquisition time of the file.

    Example
    -------
        >>> # Extract time
        >>> extract_acquisition_time(nc_file_path)
        '2019-02-02-T052350Z'
    """
    # Parse path for file name
    file, _ = os.path.splitext(os.path.basename(file_path))

    # Extract acquisition time from file
    acquisition_time = file.split("_")[-1]

    # Return acquisition time
    return acquisition_time


def extract_arrays_to_list(no2, dates):
    """Returns a list of arrays from a nested dictionary,
    that is indexed by dictionary[Year][Month][Day][Time].

    Meant for intra and inter-month date ranges (both
    continuous and not continuous).

    Parameters
    ----------
    no2 : dict
        Dictionary containing masked daily NO2 arrays,
        indexed by dictionary['YYYY']['MM']['DD']['THHMMSSZ'].

    dates : list
        List of dates (strings), formatted as 'YYYY-MM-DD'.

    Returns
    -------
    array_list : list
        List of NO2 arrays within the specified dates.

    Example
    -------
        # Create date range to extract
        >>> date_range = create_date_list('2019-02-02', '2019-02-28'),
        >>> # Get NO2 array for each date into list
        >>> no2_arrays = extract_arrays_to_list(
        ...     no2=no2_data, dates=date_range)
    """
    # Flatten dataframe into dictionary
    no2_df = json_normalize(no2)

    # Replace '.' with '-' in column names
    no2_df.columns = [column.replace(".", "-") for column in no2_df.columns]

    # Create list of arrays based on date list
    array_list = [
        no2_df[col].loc[0]
        for day in dates
        for col in no2_df.columns
        if re.compile(f"^{day}").match(col)
    ]

    # Return list of arrays
    return array_list


def store_continuous_range_statistic(
    no2_daily, date_range_list, statistic="mean"
):
    """Calculates the specified statistic for each entry
    (year/month/day) in a list of and stores the statistics
    values in a dictionary.

    Parameters
    ----------
    radiance_daily : dict
        Dictionary containing daily radiance arrays,
        indexed by radiance['YYYY']['MM']['DD'].

    date_range_list : list (of str)
        List containing strings of format 'YYYY-MM-DD'.

    Returns
    -------
    no2_date_range_statistic : dict
        Dictionary containig date range variance radiance arrays,
        indexed by radiance_date_range_statisic['YYYYMMDD-YYYYMMDD'].

    Example
    -------
        >>> # Define date ranges
        >>> feb_date_range_list = [
        ...     ('2019-02-02', '2019-02-28')
        ... ]
        >>> # Store varaiance
        >>> feb_2019_variance = store_continuous_range_statistic(
        ...     no2_daily=no2_feb_2019,
        ...     date_range_list=feb_date_range_list,
        ...     statistic='variance')
        >>> # Show keys
        >>> for key in feb_2018_variance.keys():
        ...     print(key)
        20190202-20190228
    """
    # Raise error if input radiance data is not a dictionary
    if not isinstance(no2_daily, dict):
        raise TypeError("Input data must be of type dict.")

    # Raise error if input date data is not a list
    if not isinstance(date_range_list, list):
        raise TypeError("Input data must be of type list.")

    # Create list of date ranges for start/end date combo
    date_ranges = [
        rd.create_date_list(start_date, end_date)
        for start_date, end_date in date_range_list
    ]

    # Initialize dictionary to store radiance arrays
    no2_date_range_statistic = {}

    # Loop through all months
    for date_range in date_ranges:

        # Create index based on date range
        date_key = (
            f"{date_range[0].replace('-', '')}"
            f"-{date_range[-1].replace('-', '')}"
        )

        # Get arrays for all dates into list
        no2_arrays = extract_arrays_to_list(no2=no2_daily, dates=date_range)

        # Check statistic type
        # Mean
        if statistic == "mean":

            # Get mean for each pixel, over all arrays (bands)
            no2_statistic = rd.calculate_statistic(
                no2_arrays, statistic="mean"
            )

        # Variance
        elif statistic == "variance":

            # Get variance for each pixel, over all arrays (bands)
            no2_statistic = rd.calculate_statistic(
                no2_arrays, statistic="variance"
            )

        # Standard deviation
        elif statistic == "deviation":

            # Get standard deviation for each pixel, over all arrays (bands)
            no2_statistic = rd.calculate_statistic(
                no2_arrays, statistic="deviation"
            )

        # Any other value
        else:
            raise ValueError(
                "Invalid statistic. Function supports "
                "'mean', 'variance', or 'deviation'."
            )

        # Add statistic array to dictionary
        if date_key not in no2_date_range_statistic.keys():
            no2_date_range_statistic[date_key] = no2_statistic

    # Return date range statistic
    return no2_date_range_statistic


def convert_level2_to_level3(
    level2_folder,
    import_operations,
    level3_netcdf_folder,
    level3_geotiff_folder,
    level3_prefix,
):
    """Converts Sentinel-5P Level-2 netCDF files to Level-3
    netCDF and GeoTiff files.
    """
    # Create harp import operations string
    operations = create_import_operations(**import_operations)

    # Loop through Level-2 files
    for level2_file in level2_folder:

        # Extract acquisition time
        acquisition_time = extract_acquisition_time(level2_file)

        try:
            # Convert netCDF from Level-2 to Level-3
            level3_path = resample_netcdf4(
                level2_file,
                operations,
                export_folder=level3_netcdf_folder,
                file_prefix=level3_prefix,
                acquisition_time=acquisition_time,
            )

        # Empty file (no data)
        except NoDataError as error:
            print(f"{acquisition_time} {error}")

        # Non-empty file
        else:
            # Extract Level-3 data to array
            level3_data = extract_no2_data(level3_path)

            # Create transform
            transform = extract_no2_transform(level3_path)

            # Define export metadata
            metadata = rd.create_metadata(
                array=level3_data, transform=transform, nodata=np.nan
            )

            # Define outpath
            output_path = os.path.join(
                level3_geotiff_folder,
                f"{level3_prefix}-{acquisition_time}.tif",
            )

            # Export array to GeoTiff
            rd.export_array(
                array=level3_data, output_path=output_path, metadata=metadata
            )


def store_level3_data(level3_folder):
    """Returns an array containing Level-3 data from netCDF
    files, indexed by data acquisition year, month, day,
    and time.

    Parameters
    ----------
    level3_folder : str
        Path to the the folder containing the Level-3 netCDF files.

    Returns
    -------

    Example
    -------
    """
    # Initialize dictionary for data arrays
    data_dict = {}

    # Loop through Level-3 netCDF files
    for level3_file in glob.glob(os.path.join(level3_folder, "*.nc")):

        # Store NO2 data
        store_no2_data(
            no2_data=extract_no2_data(level3_file),
            no2_dict=data_dict,
            acquisition_time=extract_acquisition_time_processed(level3_file),
        )

    # Return dictionary
    return data_dict


def plot_change(
    pre_change,
    post_change,
    location="South Korea",
    title="NO2",
    data_source="European Space Agency",
):
    """Plots two arrays and the difference
    between the arrays on the same figure.

    pre_change : numpy array
        Numpy array containing radiance values.

    post_change : numpy array
        Numpy array containing radiance values.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'Penn State Campus'.

    titles : list of str, optional
        Plot sub-titles. Default value is ['Radiance',
        'Radiance', 'Radiance']. Intended for ['September
        2019 Mean Radiance', 'March 2020 Mean Radiance',
       'Change in Mean Radiance (September 2019 vs. March
       2020)'].

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'NASA Black Marble'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>> # Define titles
        >>> plot_titles = [
        ...     'September 2019 Mean Radiance',
        ...     'March 2020 Mean Radiance',
        ...     'Change in Mean Radiance (September 2019 vs. March 2020)'
        ... ]
        >>> # Plot Sept 2019 and March 2020
        >>> fig, ax = plot_change(
        >>>     pre_change=radiance_monthtly_mean.get('2019').get('09'),
        >>>     post_change=radiance_monthtly_mean.get('2020').get('03'),
                titles=plot_titles)
    """
    # Calculate difference (post-change - pre-change)
    diff = post_change - pre_change

    # Find absolute values for change min & max
    diff_min_abs = np.absolute(diff.min())
    diff_max_abs = np.absolute(diff.max())

    # Determine max value (for plotting vmin/vmax)
    diff_vmax = diff_min_abs if (diff_min_abs > diff_max_abs) else diff_max_abs

    diff_vmin = -diff_vmax

    # Define radiance units
    units = r"$\mathrm{mol \cdot m^{-2}}$"

    # Define title
    diff_title = f"{title} ({units})"

    # Define color maps
    diff_cmap = "RdBu_r"

    # Use dark background
    with plt.style.context("dark_background"):

        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(12, 12))

        # Add super title
        plt.suptitle(f"{location} Nitrogen Dioxide Change", size=24)

        # Adjust spacing
        # plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(top=0.98)

        # Plot diff array
        ep.plot_bands(
            diff,
            scale=False,
            title=diff_title,
            vmin=diff_vmin,
            vmax=diff_vmax,
            cmap=diff_cmap,
            ax=ax,
        )

        # Add caption
        fig.text(
            0.5, 0.15, f"Data Source: {data_source}", ha="center", fontsize=16
        )

        # Set title size
        ax.title.set_size(20)

    # Return figure and axes object
    return fig, ax


def plot_change_with_boundary(
    pre_change,
    post_change,
    extent_file,
    location="South Korea",
    title="NO2",
    data_source="European Space Agency",
):
    """Plots two arrays and the difference
    between the arrays on the same figure.

    pre_change : numpy array
        Numpy array containing radiance values.

    post_change : numpy array
        Numpy array containing radiance values.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'Penn State Campus'.

    titles : list of str, optional
        Plot sub-titles. Default value is ['Radiance',
        'Radiance', 'Radiance']. Intended for ['September
        2019 Mean Radiance', 'March 2020 Mean Radiance',
       'Change in Mean Radiance (September 2019 vs. March
       2020)'].

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'NASA Black Marble'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>> # Define titles
        >>> plot_titles = [
        ...     'September 2019 Mean Radiance',
        ...     'March 2020 Mean Radiance',
        ...     'Change in Mean Radiance (September 2019 vs. March 2020)'
        ... ]
        >>> # Plot Sept 2019 and March 2020
        >>> fig, ax = plot_change(
        >>>     pre_change=radiance_monthtly_mean.get('2019').get('09'),
        >>>     post_change=radiance_monthtly_mean.get('2020').get('03'),
                titles=plot_titles)
    """
    # Get plotting extent
    with rio.open(extent_file) as src:
        src_extent = plotting_extent(src)

    # Calculate difference (post-change - pre-change)
    diff = post_change - pre_change

    # Find absolute values for change min & max
    diff_min_abs = np.absolute(diff.min())
    diff_max_abs = np.absolute(diff.max())

    # Determine max value (for plotting vmin/vmax)
    diff_vmax = diff_min_abs if (diff_min_abs > diff_max_abs) else diff_max_abs

    diff_vmin = -diff_vmax

    # Define radiance units
    units = r"$\mathrm{mol \cdot m^{-2}}$"

    # Define titles
    diff_title = f"{title} ({units})"

    # Define color maps
    diff_cmap = "RdBu_r"

    # Use dark background
    with plt.style.context("dark_background"):

        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(12, 12))

        # Add super title
        plt.suptitle(f"{location} Nitrogen Dioxide Change", size=24)

        # Adjust spacing
        # plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(top=0.98)

        # Plot diff array
        ep.plot_bands(
            diff,
            scale=False,
            title=diff_title,
            vmin=diff_vmin,
            vmax=diff_vmax,
            cmap=diff_cmap,
            ax=ax,
            extent=src_extent,
        )

        # Add caption
        fig.text(
            0.5, 0.15, f"Data Source: {data_source}", ha="center", fontsize=16
        )

        # Set title size
        ax.title.set_size(20)

    # Return figure and axes object
    return fig, ax


def plot_percent_change(
    percent_change,
    location="South Korea",
    title="NO2",
    data_source="European Space Agency",
):
    """Plots two arrays and the difference
    between the arrays on the same figure.

    pre_change : numpy array
        Numpy array containing radiance values.

    post_change : numpy array
        Numpy array containing radiance values.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'Penn State Campus'.

    titles : list of str, optional
        Plot sub-titles. Default value is ['Radiance',
        'Radiance', 'Radiance']. Intended for ['September
        2019 Mean Radiance', 'March 2020 Mean Radiance',
       'Change in Mean Radiance (September 2019 vs. March
       2020)'].

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'NASA Black Marble'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>> # Define titles
        >>> plot_titles = [
        ...     'September 2019 Mean Radiance',
        ...     'March 2020 Mean Radiance',
        ...     'Change in Mean Radiance (September 2019 vs. March 2020)'
        ... ]
        >>> # Plot Sept 2019 and March 2020
        >>> fig, ax = plot_change(
        >>>     pre_change=radiance_monthtly_mean.get('2019').get('09'),
        >>>     post_change=radiance_monthtly_mean.get('2020').get('03'),
                titles=plot_titles)
    """
    # Find absolute values for change min & max
    percent_change_min_abs = np.absolute(percent_change.min())
    percent_change_max_abs = np.absolute(percent_change.max())

    # Determine max value (for plotting vmin/vmax)
    diff_vmax = (
        percent_change_min_abs
        if (percent_change_min_abs > percent_change_max_abs)
        else percent_change_max_abs
    )

    diff_vmin = -diff_vmax

    # Define radiance units
    units = "%"

    # Define titles
    diff_title = f"{title} ({units})"

    # Define color maps
    diff_cmap = "RdBu_r"

    # Use dark background
    with plt.style.context("dark_background"):

        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(12, 12))

        # Add super title
        plt.suptitle(f"{location} Nitrogen Dioxide Change", size=24)

        # Adjust spacing
        # plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(top=0.98)

        # Plot diff array
        ep.plot_bands(
            percent_change,
            scale=False,
            title=diff_title,
            vmin=diff_vmin,
            vmax=diff_vmax,
            cmap=diff_cmap,
            ax=ax,
        )

        # Add caption
        fig.text(
            0.5, 0.15, f"Data Source: {data_source}", ha="center", fontsize=16
        )

        # Set title size
        ax.title.set_size(20)

    # Return figure and axes object
    return fig, ax


def plot_percent_change_with_boundary(
    percent_change,
    extent_file,
    location="South Korea",
    title="NO2",
    data_source="European Space Agency",
):
    """Plots two arrays and the difference
    between the arrays on the same figure.

    pre_change : numpy array
        Numpy array containing radiance values.

    post_change : numpy array
        Numpy array containing radiance values.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'Penn State Campus'.

    titles : list of str, optional
        Plot sub-titles. Default value is ['Radiance',
        'Radiance', 'Radiance']. Intended for ['September
        2019 Mean Radiance', 'March 2020 Mean Radiance',
       'Change in Mean Radiance (September 2019 vs. March
       2020)'].

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'NASA Black Marble'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>> # Define titles
        >>> plot_titles = [
        ...     'September 2019 Mean Radiance',
        ...     'March 2020 Mean Radiance',
        ...     'Change in Mean Radiance (September 2019 vs. March 2020)'
        ... ]
        >>> # Plot Sept 2019 and March 2020
        >>> fig, ax = plot_change(
        >>>     pre_change=radiance_monthtly_mean.get('2019').get('09'),
        >>>     post_change=radiance_monthtly_mean.get('2020').get('03'),
                titles=plot_titles)
    """
    # Get plotting extent
    with rio.open(extent_file) as src:
        src_extent = plotting_extent(src)

    # Find absolute values for change min & max
    percent_change_min_abs = np.absolute(percent_change.min())
    percent_change_max_abs = np.absolute(percent_change.max())

    # Determine max value (for plotting vmin/vmax)
    diff_vmax = (
        percent_change_min_abs
        if (percent_change_min_abs > percent_change_max_abs)
        else percent_change_max_abs
    )

    diff_vmin = -diff_vmax

    # Define radiance units
    units = "%"

    # Define titles
    diff_title = f"{title} ({units})"

    # Define color maps
    diff_cmap = "RdBu_r"

    # Use dark background
    with plt.style.context("dark_background"):

        # Create figure and axes objects
        fig, ax = plt.subplots(figsize=(12, 12))

        # Add super title
        plt.suptitle(f"{location} Nitrogen Dioxide Change", size=24)

        # Adjust spacing
        plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(top=0.98)

        # Plot diff array
        ep.plot_bands(
            percent_change,
            scale=False,
            title=diff_title,
            vmin=diff_vmin,
            vmax=diff_vmax,
            cmap=diff_cmap,
            ax=ax,
            extent=src_extent,
        )

        # Add caption
        fig.text(
            0.5, 0.15, f"Data Source: {data_source}", ha="center", fontsize=16
        )

        # Set title size
        ax.title.set_size(20)

    # Return figure and axes object
    return fig, ax


def plot_histogram(
    radiance,
    location="South Korea",
    title="Distribution of Percent Change, Jan-Jun Mean NO2, 2019-2020",
    xlabel="Percent Change",
    ylabel="Pixel Count",
    data_source="European Space Agency",
    difference=True,
):
    """Plots the distribution of values in a radiance array.

    Parameters
    ----------
    radiance : numpy array
        Array containing raw values, mean values,
        or difference values.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'Penn State Campus'.

    title : str, optional
        Plot sub-title. Default value is 'Distribution of
        Radiance'. Intended for 'Distribution of September
        2019 Mean Radiance' or 'Distribution of the Change
        in Mean Radiance (September 2019 vs. March 2020).'

    xlabel : str, optional
        Label on the x-axis. Default value is 'Radiance'.

    ylabel : str, optional
        Label on the y-axis. Default value is 'Pixel Count'.

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'NASA Black Marble'.

    difference : bool, optional
        Boolean indicating if the array contains raw
        values or mean values (False) or contains
        difference values (True). Default value is False.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot object
            The axes object associated with the histogram.

    Example
    -------
        >>> # Plot Sept 2019 vs. March 2020 change histogram
        >>> fig, ax = plot_histogram(
        ...     diff_sep_2019_march_2020,
        ...     title="Distribution of the Change in Mean Radiance",
        ...     xlabel='Change in Mean Radiance',
        ...     difference=True)
    """
    # Find absolute values for radiance min & max
    radiance_min_abs = np.absolute(radiance.min())
    radiance_max_abs = np.absolute(radiance.max())

    # Determine max value (for plotting vmin/vmax)
    plot_max = (
        radiance_min_abs
        if (radiance_min_abs > radiance_max_abs)
        else radiance_max_abs
    )

    # Define vmin and vmax
    hist_min = -plot_max if difference else 0
    hist_max = plot_max

    # Define histogram range
    hist_range = (hist_min, hist_max)

    # Define radiance units
    units = "%"

    # Use dark background
    with plt.style.context("dark_background"):

        # Create figure and axes object
        fig, ax = ep.hist(
            radiance,
            hist_range=hist_range,
            colors="#984ea3",
            title=title,
            xlabel=f"{xlabel} ({units})",
            ylabel=ylabel,
        )

        # Add super title
        plt.suptitle(f"{location} Nitrogen Dioxide", size=24)

        # Adjust spacing
        plt.subplots_adjust(top=0.9)

        # Add caption
        fig.text(
            0.5, 0.03, f"Data Source: {data_source}", ha="center", fontsize=14
        )

    # Return figure and axes object
    return fig, ax


def plot_normalized_histogram(
    data,
    location="South Korea",
    title="Distribution of Percent Masked, Jul 2018 - Jul 2020",
    xlabel="Percent Masked (%)",
    ylabel="Normalized Pixel Count (Probability Density)",
    data_source="European Space Agency",
):
    """Plots the distribution of data, normalized to a probability density.

    Parameters
    ----------
    data : numpy array
        Array containing to plot.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'South Korea'.

    title : str, optional
        Plot sub-title. Default value is 'Distribution of Percent Masked,
        Jul 2018 - Jul 2020'.

    xlabel : str, optional
        Label on the x-axis. Default value is 'Percent Masked (%)'.

    ylabel : str, optional
        Label on the y-axis. Default value is 'Normalized Pixel Count
        (Probability Density)'.

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'European Space Agency'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot object
            The axes object associated with the histogram.

    Example
    -------
        >>> # Plot normalized histogram
        >>> fig, ax = plot_normalized_histogram(percent_masked_arr)
    """
    # Plot histogram normalized to form a probability density
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.hist(data, density=True, bins=20, color="#984ea3")
        plt.suptitle(f"{location} Nitrogen Dioxide", size=24)
        plt.subplots_adjust(top=0.9)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.text(
            0.5, 0.03, f"Data Source: {data_source}", ha="center", fontsize=14
        )

    return fig, ax


def extract_plotting_extent(raster_path):
    """Extracts the plotting extent from a raster.

    Parameters
    ----------
    raster_path : str
        Path to the raster file.

    Returns
    -------
    raster_extent : tuple
        Tuple containing (longitude min, longitude max,
        latitude min, latitude max).

    Example
    -------
        >>> # Extract plotting extent
        >>> extract_plotting_extent(
        ...  "S5P-OFFL-L3-NO2-20190101-20190630-MEAN-MOL-PER-M2.tif"
        ... )
        (125.0, 131.0, 33.1, 38.7)
    """
    # Raise error for invalid file path
    if not os.path.exists(raster_path):
        raise ValueError("Invalid raster file path.")

    # Get plotting extent
    with rio.open(raster_path) as src:
        raster_extent = plotting_extent(src)

    return raster_extent


def project_vector(vector_path, raster_path):
    """Projects a vector to match a raster CRS if the two differ.

    Parameters
    ----------
    vector_path : str
        Path to the vector file. The file that will be projected.

    raster_path : str
        Path to the raster file. The file that has the CRS to which
        the vector will be projected.

    Returns
    -------
    vector_projected : geopandas geodataframe
        Geodataframe of the vector in the same CRS as the raster file.

    Example
    -------

    """
    # Raise error for invalid file path
    if not os.path.exists(vector_path):
        raise ValueError("Invalid vector file path.")
    if not os.path.exists(raster_path):
        raise ValueError("Invalid raster file path.")

    # Project vector to raster CRS if the two do not match
    vector = gpd.read_file(vector_path)
    with rio.open(raster_path) as src:
        vector_projected = (
            vector.to_crs(src.crs) if vector.crs != src.crs else vector
        )

    return vector_projected


def create_polygon_from_extent(extent):
    """Creates a Shapely polygon from a bounding box extent.

    Parameters
    ----------
    extent : tuple
        Tuple containing (longitude min, longitude max,
        latitude min, latitude max).

    Returns
    -------
    polygon : shapely.geometry.polygon.Polygon

    Example
    -------

    """
    # Raise errors for invalid extent values
    if not isinstance(extent, tuple):
        raise TypeError("Extent must be a tuple")
    if len(extent) != 4:
        raise ValueError(
            "Extent must have 4 values: (lon_min, lon_max, lat_min, lat_max)."
        )
    if extent[0] == extent[1]:
        raise ValueError("Extent longitude min/max must differ.")
    if extent[2] == extent[3]:
        raise ValueError("Extent latitude min/max must differ.")
    if (extent[0] >= extent[1]) or (extent[2] >= extent[3]):
        raise ValueError(
            "Extent order must be: (lon_min, lon_max, lat_min, lat_max)."
        )
    if extent[0] < -180:
        raise ValueError("Minimum longitude must be >= -180.")
    if extent[1] > 180:
        raise ValueError("Maximum longitude must be <= 180.")
    if extent[2] < -90:
        raise ValueError("Minimum latitude must be >= -90.")
    if extent[3] > 90:
        raise ValueError("Maximum latitude must be <= 90.")

    # Create polygon from extent
    polygon = Polygon(
        [
            (extent[0], extent[2]),
            (extent[0], extent[3]),
            (extent[1], extent[3]),
            (extent[1], extent[2]),
            (extent[0], extent[2]),
        ]
    )

    return polygon


def read_geotiff_into_array(geotiff_path, dimensions=1):
    """Reads a GeoTif file into a Numpy array.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTiff file.

    Returns
    -------
    array : numpy array
        Array containing the data.

    Example
    -------

    """
    # Read-in array
    with rio.open(geotiff_path) as file:
        array = file.read(dimensions)

    return array


def extract_geotiff_metadata(geotiff_path):
    """Extracts the metadata from a GeoTiff file.

    Parameters
    ----------
    geotiff_path : str
        Path to the GeoTiff file.

    Returns
    -------
    metadata : dict
        Dictionary containing the metadata.

    Example
    -------

    """
    # Extract metadata
    with rio.open(geotiff_path) as file:
        metadata = file.meta

    return metadata


def calculate_magnitude_change(pre_change, post_change):
    """Calculates the magnitude change from the pre-change data
    to the post-change data (post-change minus pre-change).

    Parameters
    ----------
    pre_change : numpy array
        Array containing the pre-change data.

    post_change : numpy array
        Array containing the post-change data.

    Returns
    -------
    magnitude_change : numpy array
        Array containing the magnitude difference between
        the post-change and pre-change arrays.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Raise error if input data are not arrays
    if not (
        isinstance(pre_change, np.ndarray)
        or isinstance(post_change, np.ndarray)
    ):
        raise TypeError("Input data must be of type array.")

    # Calculate magnitude change
    magnitude_change = post_change - pre_change

    return magnitude_change


def calculate_percent_change(pre_change, post_change):
    """Calculates the percent change from the pre-change data
    to the post-change data (post-change minus pre-change).
    Parameters
    ----------
    pre_change : numpy array
        Array containing the pre-change data.

    post_change : numpy array
        Array containing the post-change data.

    Returns
    -------
    percent_change : numpy array
        Array containing the percent difference between
        the post-change and pre-change arrays.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Raise error if input data are not arrays
    if not (
        isinstance(pre_change, np.ndarray)
        or isinstance(post_change, np.ndarray)
    ):
        raise TypeError("Input data must be of type array.")

    # Calculate percent change
    percent_change = (
        np.divide(
            (post_change - pre_change), pre_change, where=(pre_change != 0)
        )
        * 100
    )

    return percent_change


def plot_monthly_comparison(
    pre_change,
    post_change,
    extent_file,
    country_boundaries,
    country_names,
    change_type="magnitude",
    location="South Korea",
    titles=["Subplot 1", "Subplot 2", "Subplot 3"],
    data_source="European Space Agency",
):
    """Plots a comparison between the mean NO2 for two different months.

    Parameters
    ----------
    pre_change : numpy array
        Array containing pre-change data.

    post_change : numpy array
        Array containing post-change data.

    extent_file : str
        Path to a GeoTiff file from which to extract the plotting extent
        for the data.

    country_boundaries : list of geopandas geodataframes
        List containing the country boundaries to add.

    country_names : list of str
        List containing the country names (for labeling plots).

    change_type : str, optional
        Type of change to display for in the third subplot. Valid options
        are 'magnitude' and 'percent'. Default value is 'magnitude'.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'South Korea'.

    titles : list of str, optional
        Subplot titles.
        Default value is ['Subplot 1', 'Subplot 2', 'Subplot 3'].
        Ex: ['May 2019 Mean NO2', 'May 2020 Mean NO2', 'Change in Mean NO2'].

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'European Space Agency'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Raise error for invalid change type
    if change_type.lower() not in ["magnitude", "percent"]:
        raise ValueError("Change type must be 'magnitude' or 'percent'.")

    # Extract magnitude min/max for plotting (from pre- and post-change data)
    magnitude_vmax = np.array([pre_change.max(), post_change.max()]).max()
    magnitude_vmin = 0

    # Calculate change
    change = (
        calculate_magnitude_change(
            pre_change=pre_change, post_change=post_change
        )
        if change_type == "magnitude"
        else calculate_percent_change(
            pre_change=pre_change, post_change=post_change
        )
    )

    # Extract change min/max (for plotting vmin/vmax)
    change_vmax = (
        np.absolute(change.min())
        if (np.absolute(change.min()) > np.absolute(change.max()))
        else np.absolute(change.max())
    )
    change_vmin = -change_vmax

    # Get plotting extent
    extent = extract_plotting_extent(extent_file)

    # Define plot settings
    magnitude_cmap = "inferno"
    change_cmap = "RdBu_r"

    # Plot data
    with plt.style.context("dark_background"):
        # Initialize figure/axes
        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        plt.suptitle(f"{location} Nitrogen Dioxide", size=24)
        plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(top=0.98)

        # Subplot 1 (pre-change)
        ep.plot_bands(
            pre_change,
            scale=False,
            title=titles[0],
            vmin=magnitude_vmin,
            vmax=magnitude_vmax,
            cmap=magnitude_cmap,
            ax=ax[0],
            extent=extent,
        )

        # Subplot 2 (post-change)
        ep.plot_bands(
            post_change,
            scale=False,
            title=titles[1],
            vmin=magnitude_vmin,
            vmax=magnitude_vmax,
            cmap=magnitude_cmap,
            ax=ax[1],
            extent=extent,
        )

        # Subplot 3 (change)
        ep.plot_bands(
            change,
            scale=False,
            title=titles[2],
            vmin=change_vmin,
            vmax=change_vmax,
            cmap=change_cmap,
            ax=ax[2],
            extent=extent,
        )

        # Add country boundaries and legend to axes
        for axis in ax:
            country_boundaries[0].boundary.plot(
                edgecolor="#e41a1c",
                linewidth=0.5,
                ax=axis,
                alpha=1,
                label=country_names[0],
            )
            country_boundaries[1].boundary.plot(
                edgecolor="#1b7837",
                linewidth=0.5,
                ax=axis,
                alpha=1,
                label=country_names[1],
            )
            axis.legend(
                shadow=True, edgecolor="white", fontsize=10, loc="lower right"
            )

        # Add caption
        fig.text(
            0.5, 0.15, f"Data Source: {data_source}", ha="center", fontsize=16
        )

    return fig, ax


def plot_monthly_comparison_limit_scale(
    pre_change,
    post_change,
    extent_file,
    country_boundaries,
    country_names,
    change_type="magnitude",
    location="South Korea",
    titles=["Subplot 1", "Subplot 2", "Subplot 3"],
    data_source="European Space Agency",
):
    """Plots a comparison between the mean NO2 for two different months.

    Parameters
    ----------
    pre_change : numpy array
        Array containing pre-change data.

    post_change : numpy array
        Array containing post-change data.

    extent_file : str
        Path to a GeoTiff file from which to extract the plotting extent
        for the data.

    country_boundaries : list of geopandas geodataframes
        List containing the country boundaries to add.

    country_names : list of str
        List containing the country names (for labeling plots).

    change_type : str, optional
        Type of change to display for in the third subplot. Valid options
        are 'magnitude' and 'percent'. Default value is 'magnitude'.

    location : str, optional
        Name of study area location. Included in plot
        super-title. Default value is 'South Korea'.

    titles : list of str, optional
        Subplot titles.
        Default value is ['Subplot 1', 'Subplot 2', 'Subplot 3'].
        Ex: ['May 2019 Mean NO2', 'May 2020 Mean NO2', 'Change in Mean NO2'].

    data_source : str, optional
        Sources of data used in the plot.
        Default value is 'European Space Agency'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the histogram.

        ax : matplotlib.axes._subplots.AxesSubplot objects
            The axes objects associated with the histogram.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Raise error for invalid change type
    if change_type.lower() not in ["magnitude", "percent"]:
        raise ValueError("Change type must be 'magnitude' or 'percent'.")

    # Extract magnitude min/max for plotting (from pre- and post-change data)
    magnitude_vmax = np.array([pre_change.max(), post_change.max()]).max()
    magnitude_vmin = 0

    # Calculate change
    change = (
        calculate_magnitude_change(
            pre_change=pre_change, post_change=post_change
        )
        if change_type == "magnitude"
        else calculate_percent_change(
            pre_change=pre_change, post_change=post_change
        )
    )

    # Extract change min/max (for plotting vmin/vmax)
    # Limit to +/- 100 for percent, to keep a consistent scale
    change_vmax = (
        (
            np.absolute(change.min())
            if (np.absolute(change.min()) > np.absolute(change.max()))
            else np.absolute(change.max())
        )
        if change_type == "magnitude"
        else 100
    )
    change_vmin = -change_vmax

    # Get plotting extent
    extent = extract_plotting_extent(extent_file)

    # Define plot settings
    magnitude_cmap = "inferno"
    change_cmap = "RdBu_r"

    # Plot data
    with plt.style.context("dark_background"):
        # Initialize figure/axes
        fig, ax = plt.subplots(1, 3, figsize=(24, 8))
        plt.suptitle(f"{location} Nitrogen Dioxide", size=24)
        plt.subplots_adjust(hspace=0.15)
        plt.subplots_adjust(top=0.98)

        # Subplot 1 (pre-change)
        ep.plot_bands(
            pre_change,
            scale=False,
            title=titles[0],
            vmin=magnitude_vmin,
            vmax=magnitude_vmax,
            cmap=magnitude_cmap,
            ax=ax[0],
            extent=extent,
        )

        # Subplot 2 (post-change)
        ep.plot_bands(
            post_change,
            scale=False,
            title=titles[1],
            vmin=magnitude_vmin,
            vmax=magnitude_vmax,
            cmap=magnitude_cmap,
            ax=ax[1],
            extent=extent,
        )

        # Subplot 3 (change)
        ep.plot_bands(
            change,
            scale=False,
            title=titles[2],
            vmin=change_vmin,
            vmax=change_vmax,
            cmap=change_cmap,
            ax=ax[2],
            extent=extent,
        )

        # Add country boundaries and legend to axes
        for axis in ax:
            country_boundaries[0].boundary.plot(
                edgecolor="#e41a1c",
                linewidth=0.5,
                ax=axis,
                alpha=1,
                label=country_names[0],
            )
            country_boundaries[1].boundary.plot(
                edgecolor="#1b7837",
                linewidth=0.5,
                ax=axis,
                alpha=1,
                label=country_names[1],
            )
            axis.legend(
                shadow=True, edgecolor="white", fontsize=10, loc="lower right"
            )

        # Add caption
        fig.text(
            0.5, 0.15, f"Data Source: {data_source}", ha="center", fontsize=16
        )

    return fig, ax


def aggregate_raster_data(
    raster_path, vector_path, zonal_statistics="count sum"
):
    """Aggregates raster data to vector polygons, based on specified
    aggregation metrics.

    Parameters
    ----------
    raster_path : str
        Path to raster file containing data that will be aggregated.

    vector_path : str
        Path to the vector file containing polygons to which data will
        be aggregated.

    zonal_statistics : space-delimited str, optional
        Zonal statistics to calculate. Default value is 'count sum'.

    Returns
    -------
    aggregated_data : geopandas geodataframe
        Geodataframe containing the raster data aggreagated with the
        vector polygons.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Extract data and metadata from raster
    with rio.open(raster_path) as src:
        data = src.read(1, masked=True)
        metadata = src.profile

    # Extract zonal stats into geodataframe
    aggregated_data = gpd.GeoDataFrame.from_features(
        rs.zonal_stats(
            vectors=gpd.read_file(vector_path),
            raster=data,
            nodata=metadata.get("nodata"),
            affine=metadata.get("transform"),
            geojson_out=True,
            copy_properties=True,
            stats=zonal_statistics,
        )
    )

    return aggregated_data


def magnitude_change(pre_change, post_change):
    """Calculates the magnitude change."""
    # Calculate magnitude change
    change = post_change - pre_change

    return change


def percent_change(pre_change, post_change):
    """Calculates the percent change."""
    # Calculate percent change
    change = (
        (post_change - pre_change) / pre_change * 100 if pre_change != 0 else 0
    )

    return change


def get_geometry(shapefile_path, geometry_column="geometry"):
    """Returns a geodataframe with only the index and geometry columns."""
    # Get geodataframe with only geometry column
    geometry = gpd.read_file(shapefile_path)[[geometry_column]]

    return geometry


def clean_data(geodatframe, new_name):
    """Creates a new dataframe with only the mean data and renames
    the mean column to a specified name.
    """
    # Create new dataframe with mean data and rename column
    cleaned_data = geodatframe[["mean"]].rename(
        columns={"mean": new_name}, copy=True
    )

    return cleaned_data


def add_change(
    dataframe,
    pre_change_column,
    post_change_column,
    new_column,
    change_type="magnitude",
):
    """Calculates and adds a magnitude or percent change column
    to all dataframe rows, based on two input columns.
    """
    # Calculate and add change
    dataframe[new_column] = (
        dataframe.apply(
            lambda row: magnitude_change(
                pre_change=row[pre_change_column],
                post_change=row[post_change_column],
            ),
            axis=1,
        )
        if change_type == "magnitude"
        else dataframe.apply(
            lambda row: percent_change(
                pre_change=row[pre_change_column],
                post_change=row[post_change_column],
            ),
            axis=1,
        )
    )

    # Set output message
    message = print(f"Added new column: {new_column}")

    return message


def plot_aggregate_change(
    aggregated_data,
    magnitude_change,
    percent_change,
    vector_boundary,
    plot_vector=False,
    super_title="Nitrogen Dioxide Change",
    data_source="European Space Agency",
    figsize=(20, 8),
):
    """Plots the NO2 change aggregated a different levels.

    Parameters
    ----------
    aggregated_data : geopandas geodataframe
        Geodataframe containing the aggregated data.

    magnitude_change : str
        Column name for the magnitude change.

    percent_change : str
        Column name for the percent change.

    vector_boundary : geopandas geodataframe
        Geodataframe containing a vector layer to be overlayed.

    plot_vector : bool, optional
        Whether or not to plot the vector layer. Default value is False.

    super_title : str, optional
        Main plot title. Default value is 'Nitrogen Dioxide Change'.

    data_source : str, optional
        Data source. Default value is 'European Space Agency'

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the plot.

        ax : matplotlib.axes._subplots.AxesSubplot object
            The axes object associated with the plot.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Determine magnitude plot scale
    magnitude_max_absolute = abs(aggregated_data[magnitude_change].max())
    magnitude_min_absolute = abs(aggregated_data[magnitude_change].min())
    magnitude_vmax = (
        magnitude_max_absolute
        if magnitude_max_absolute > magnitude_min_absolute
        else magnitude_min_absolute
    )

    # Plot magnitude and percent change
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        plt.suptitle(super_title, size=24)

        # Set legend options
        divider_magnitude = make_axes_locatable(ax[0])
        divider_percent = make_axes_locatable(ax[1])
        cax_magnitude = divider_magnitude.append_axes(
            "right", size="5%", pad=-0.25
        )
        cax_percent = divider_percent.append_axes(
            "right", size="5%", pad=-0.25
        )

        # Magnitude change
        aggregated_data.plot(
            column=magnitude_change,
            ax=ax[0],
            legend=True,
            cax=cax_magnitude,
            vmax=magnitude_vmax,
            vmin=-magnitude_vmax,
            cmap="RdBu_r",
            linewidth=0.25,
            edgecolor="gray",
        )

        # Plot vector boundary
        if plot_vector:
            vector_boundary.plot(
                ax=ax[0], facecolor="None", edgecolor="black", linewidth=0.5
            )

        # Configure axes
        ax[0].set_title("Magnitude Change")
        ax[0].set_xlabel("Longitude (degrees)")
        ax[0].set_ylabel("Latitude (degrees)")

        # Percent change
        aggregated_data.plot(
            column=percent_change,
            ax=ax[1],
            legend=True,
            cax=cax_percent,
            cmap="RdBu_r",
            vmin=-100,
            vmax=100,
            linewidth=0.25,
            edgecolor="gray",
        )

        # Plot vector boundary
        if plot_vector:
            vector_boundary.plot(
                ax=ax[1],
                facecolor="None",
                edgecolor="black",
                linewidth=0.5,
            )

        # Configure axes
        ax[1].set_title("Percent Change")
        ax[1].set_xlabel("Longitude (degrees)")
        ax[1].set_ylabel("Latitude (degrees)")

        # Add caption
        fig.text(
            0.5,
            0,
            f"Data Source: {data_source}",
            ha="center",
            fontsize=16,
        )

    return fig, ax


def clean_time_series(time_series_path):
    """Prepares time series data for plotting.

    Parameters
    ----------
    time_series_path : str
        Path to the time series data (geopackage, .gpkg file).

    Returns
    -------
    cleaned_time_series : pandas dataframe
        Cleaned dataframe containing plot-ready data.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Get time series data into dataframe
    time_series = pd.DataFrame(
        gpd.read_file(time_series_path).drop(columns=["geometry"])
    ).set_index(keys="GRID_ID", drop=True)

    # Remove index name
    time_series.index.name = None

    # Transpose; get timestamps as rows and grid ids as columns
    time_series_transposed = time_series.transpose()

    # Set new index name
    time_series_transposed.index.name = "date"

    # Convert index values from string to datetime objects (for plotting)
    time_series_transposed.index = pd.to_datetime(time_series_transposed.index)

    # Create copy of transposed dataframe (to return)
    cleaned_time_series = time_series_transposed.copy()

    return cleaned_time_series


def plot_no2_time_series(
    time_series,
    grid_id,
    data_location="South Korea",
    data_source="European Space Agency",
    add_study_area_max=False,
):
    """Plots a time series for an aggregated hexagon grid.

    Parameters
    ----------
    time_series : pandas dataframe
        Dataframe containing the time series data.

    grid_id : str
        Grid ID to plot.

    data_location : str, optional
        Location of the data. Default value is 'South Korea'.

    data_source : str, optional
        Location of the data. Default value is 'European Space Agency'.

    add_study_area_max : bool, optional
        Boolean to add the study area maximum to the figure. Default value is
        False.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the plot.

        ax : matplotlib.axes._subplots.AxesSubplot object
            The axes object associated with the plot.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Compute mean and standard deviation for the grid cell
    mean = time_series.describe()[[grid_id]].loc["mean"][0]
    standard_deviation = time_series.describe()[[grid_id]].loc["std"][0]

    # Plot NO2 time series
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(24, 8))

        # Add NO2 data, mean, mean + stddev, mean - stddev, and maximum lines
        plt.scatter(
            time_series.index,
            time_series[[grid_id]],
            color="#ff7f00",
            zorder=4,
        )
        ax.axhline(
            mean,
            color="#4daf4a",
            label="Grid Cell Mean",
            linewidth=2,
        )
        ax.axhline(
            mean + standard_deviation,
            color="#984ea3",
            label="1 Standard Deviation",
            linewidth=2,
        )
        ax.axhline(
            mean - standard_deviation,
            color="#984ea3",
            linewidth=2,
        )
        if add_study_area_max:
            ax.axhline(
                time_series.stack().max(),
                color="#e41a1c",
                label="Study Area Maximum",
                linewidth=2,
            )

        # Configure axes, legend, caption
        ax.set_title(
            f"NO2 Time Series, {data_location}, Hexagon {grid_id}", fontsize=24
        )
        ax.set_xlabel("Date", fontsize=20)
        ax.set_ylabel(r"NO2 ($\mathrm{mol \cdot m^{-2}}$)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.legend(
            shadow=True, edgecolor="white", fontsize=20, loc="upper right"
        )
        fig.text(
            0.5,
            -0.025,
            f"Data Source: {data_source}",
            ha="center",
            fontsize=20,
        )

    return fig, ax


def convert_dataframe_column_to_array(dataframe_column, output_dtype=None):
    """Converts a Pandas dataframe column to a 1D NumPy array."""
    # Convert datafram column to array
    datatype = output_dtype if output_dtype else dataframe_column.dtype
    array = (
        np.array(object=dataframe_column, dtype=datatype, copy=True)
        .transpose()
        .ravel()
    )

    return array


def plot_spline(
    time_series,
    grid_id,
    spline_start,
    spline_end,
    plot_start,
    plot_end,
    add_study_area_max=False,
    add_grid_cell_max=False,
    data_location="South Korea",
    data_source="European Space Agency",
):
    """Creates a cubic spline plot for aggregated NO2 time series data.

    Parameters
    ----------
    time_series : pandas dataframe
        Dataframe containing the aggregated NO2 time series data.

    grid_id : str
        ID for the grid cell to compute the spline for and plot.

    spline_start : str
        Start date for the spline computation.

    spline_end : str
        End date for the spline computation.

    plot_start : str
        Start date for the plot.

    plot_end : str
        End date for the plot.

    add_study_area_max : bool, optional
        Boolean to add the study area time series maximum to the figure. This
        is the maximum aggregated mean over all grid IDs/cells in the study
        area time series. Default value is False.

    add_grid_cell_max : bool, optional
        Boolean to add the grid cell time series maximum to the figure. This
        is the maximum aggregated mean over the single grid ID/cell chosen for
        the plot. Default value is False.

    data_location : str, optional
        Location of the data. Default value is 'South Korea'.

    data_source : str, optional
        Location of the data. Default value is 'European Space Agency'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the plot.

        ax : matplotlib.axes._subplots.AxesSubplot object
            The axes object associated with the plot.

        cubic_spline : scipy.interpolate.fitpack2.LSQUnivariateSpline object
            Spline from the input data.

    Example
    ------
        >>>
        >>>
        >>>
        >>>
    """
    # SPLINE COMPUTATION
    # Subset original time series to spline start and end dates
    time_series_subset_spline = time_series[[grid_id]].loc[
        spline_start:spline_end
    ]

    # Get dates and NO2 values into arrays
    original_dates = convert_dataframe_column_to_array(
        dataframe_column=time_series_subset_spline.index,
        output_dtype="datetime64[ns]",
    )
    original_dates_as_float = original_dates.astype("float")
    original_no2 = convert_dataframe_column_to_array(
        dataframe_column=time_series_subset_spline[grid_id]
    )

    # Set spline weights; fill NO2 NAN values with 0.0
    spline_weights = ~np.isnan(original_no2)
    original_no2_filled = np.copy(original_no2)
    original_no2_filled[~spline_weights] = 0.0

    # Create new dates for spline interpolation (hourly frequency)
    spline_dates_computation = np.array(
        pd.date_range(
            start=original_dates[0], end=original_dates[-1], freq="1H"
        )
    )
    spline_dates_computation_as_float = spline_dates_computation.astype(
        "float"
    )

    # Create spline function
    cubic_spline = UnivariateSpline(
        x=original_dates_as_float,
        y=original_no2_filled,
        w=spline_weights,
        k=3,
    )

    # Apply spline function to hourly date frequency
    spline_no2 = cubic_spline(spline_dates_computation_as_float)

    # PLOTTING
    # Subset original time series to plot start and end dates
    time_series_subset_plot = time_series[[grid_id]].loc[plot_start:plot_end]

    # Get arrays for plotting - original data
    plot_dates = convert_dataframe_column_to_array(
        dataframe_column=time_series_subset_plot.index,
        output_dtype="datetime64[ns]",
    )
    plot_no2 = convert_dataframe_column_to_array(
        dataframe_column=time_series_subset_plot[grid_id]
    )

    # Create dataframe with spline values
    spline_subset_plot = pd.DataFrame(
        data=spline_no2, index=spline_dates_computation, columns=[grid_id]
    ).loc[plot_start:plot_end]

    # Get arrays for plotting - spline
    spline_dates_plot = convert_dataframe_column_to_array(
        dataframe_column=spline_subset_plot.index,
        output_dtype="datetime64[ns]",
    )
    spline_no2_plot = convert_dataframe_column_to_array(
        dataframe_column=spline_subset_plot[grid_id]
    )

    # Plot original and spline data
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot(
            plot_dates,
            plot_no2,
            "ro",
            markersize=5,
            color="#ff7f00",
            label="Mean Aggregated NO2",
        )
        plt.plot_date(
            x=spline_dates_plot,
            y=spline_no2_plot,
            markersize=2.5,
            color="#4daf4a",
            label="Cubic Smoothing Spline",
        )

        # Add time series maximum values
        if add_study_area_max:
            ax.axhline(
                time_series.stack().max(),
                color="#e41a1c",
                label="Study Area Maximum",
                linewidth=2,
                zorder=1,
            )

        if add_grid_cell_max:
            ax.axhline(
                time_series[[grid_id]].max()[0],
                color="#984ea3",  # 377eb8
                label="Grid ID Maximum",
                linewidth=2,
                zorder=1,
            )

        # Configure axes, legend, caption
        title_top = f"NO2 Time Series, {data_location}, Grid {grid_id}"
        title_middle = (
            f"Spline Dates: {reformat_date(spline_start)} - "
            f"{reformat_date(spline_end)}"
        )
        title_bottom = (
            f"Plot Dates:     {reformat_date(plot_start)} - "
            f"{reformat_date(plot_end)}"
        )
        ax.set_title(
            f"{title_top}\n{title_middle}\n{title_bottom}",
            fontsize=24,
        )
        ax.set_xlabel("Date", fontsize=20)
        ax.set_ylabel(r"NO2 ($\mathrm{mol \cdot m^{-2}}$)", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.legend(
            shadow=True, edgecolor="white", fontsize=20, loc="upper right"
        )
        fig.text(
            0.5,
            0.02,
            f"Data Source: {data_source}",
            ha="center",
            fontsize=16,
        )

    return fig, ax, cubic_spline


def reformat_date(date):
    """Reformats a date from YYYY-MM-DD to MM/DD/YYYY.

    Parameters
    ----------
    date : str
        Date to reformat. Must be in the following form: YYYY-MM-DD.

    Returns
    ------
    date_reformatted : str
        Reformatted date, in the following form: MM/DD/YYYY.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Reformat date to MM/DD/YYYY
    date_reformatted = f"{date[5:7]}/{date[8:]}/{date[:4]}"

    return date_reformatted


def get_spline_details(spline):
    """Returns the spline coefficients, knots, and residual.

    Parameters
    ----------
    spline : scipy.interpolate.fitpack2.LSQUnivariateSpline object
        Spline from which to get details.

    Returns
    -------
    tuple

        coefficients : numpy.ndarray of numpy.float64 objects
            Array containing the spline coefficients.

        knots : numpy.ndarray of numpy.datetime64 objects
            Array containing the spline knots.

        residual : float
            Spline residual.

    Example
    -------
        >>>
        >>>
        >>>
        >>>
    """
    # Get coefficients, knots, and residual
    coefficients = spline.get_coeffs()
    knots = spline.get_knots().astype("datetime64[ns]")
    residual = spline.get_residual()

    return coefficients, knots, residual


def save_figure(output_path):
    """Saves the current figure to a specified location.

    Parameters
    ----------
    output_path : str
        Path (including file name and extension)
        for the output file.

    Returns
    -------
    message : str
        Message indicating location of saved file
        (upon success) or error message (upon failure)/

    Example
    -------
    >>> # Set output path sand save figure
    >>> outpath = os.path.join("04-graphics-outputs", "figure.png")
    >>> save_figure(outpath)
    Saved plot: 04-graphics-outputs\\figure.png
    """
    # Save figure
    try:
        plt.savefig(
            fname=output_path, facecolor="k", dpi=300, bbox_inches="tight"
        )
    except Exception as error:
        message = print(f"Failed to save plot: {error}")
    else:
        message = print(f"Saved plot: {os.path.split(output_path)[-1]}")

    # Return message
    return message


def calculate_deltas(time_series, grid_id, max_difference=30):
    """Creates lists of timestamp and NO2 differences for measurements
    within 30 hours.

    Parameters
    ----------
    time_series : pandas dataframe
        Dataframe containing the time series data.

    grid_id : str
        Grid ID to plot.

    Returns
    -------
    tuple
        time_deltas, no2_deltas : list
            Lists containing the time and NO2 delta values.

    Example
    -------
    >>>
    >>>
    >>>
    >>>
    """
    # Initialize returns
    count = 0
    nan_count = 0
    time_deltas = []
    no2_deltas = []

    # Compare timestamps
    for time_0 in time_series[[grid_id]].index:
        for time_1 in time_series[[grid_id]].index:
            # Calculate timestamp difference
            time_difference = time_1 - time_0
            # Run calculations for timestamp differences within a
            #  specified duration
            if (time_difference <= timedelta(hours=max_difference)) and (
                time_difference > timedelta(hours=0)
            ):
                # Extract NO2 values
                time_0_no2 = time_series.loc[time_0][0]
                time_1_no2 = time_series.loc[time_1][0]
                # Add timestamp and NO2 deltas to lists (excluding NO2
                #  measurements with NaN values for time_0 or time_1)
                if np.isnan(time_0_no2) or np.isnan(time_1_no2):
                    nan_count += 1
                else:
                    # Add deltas to lists
                    time_deltas.append(time_difference)
                    no2_deltas.append(np.absolute(time_1_no2 - time_0_no2))
                    count += 1

    # Set output messages
    print(
        (
            f"Calculated deltas for {count}/{count + nan_count} "
            f"timestamp pairs within {max_difference} hours."
        )
    )
    print(
        (
            f"Skipped calculations (NaN values) for "
            f"{nan_count}/{count + nan_count} timestamp pairs within "
            f"{max_difference} hours."
        )
    )

    return time_deltas, no2_deltas


def convert_time_to_hours(time):
    """Converts a Timedelta to hours.

    Parameters
    ----------
    time : pandas._libs.tslibs.timedeltas.Timedelta objects
        Timedetla.

    Returns
    -------
    hours : float
        Timedelts in hours.

    Example
    -------
    >>>
    >>>
    >>>
    >>>
    """
    # Convert to hours
    hours = time.total_seconds() / 3600

    return hours


def format_time_delta(time_delta):
    """Formats timestamp deltas for plotting.

    Parameters
    ----------
    time_delta : list
        List containing the time values.

    Returns
    -------
    time_arr : numpy array
        Array containing the time delta values, converted to hours.

    Example
    -------
    >>>
    >>>
    >>>
    >>>
    """
    # Convert time array
    time_arr = np.array([convert_time_to_hours(time) for time in time_delta])

    return time_arr


def format_no2_delta(no2_delta):
    """Formats NO2 deltas for plotting.

    Parameters
    ----------
    no2_delta : list
        List containing the NO2 delta values.

    Returns
    -------
    no2_array : numpy array
        Array containing the NO2 delta values.

    Example
    -------
    >>>
    >>>
    >>>
    >>>
    """
    # Convert NO2 to array
    no2_arr = np.array(no2_delta)

    return no2_arr


def extract_grid_statistic(time_series, grid_id, statistic_type="mean"):
    """Extract a specified statistic from an NO2 grid cell.

    Parameters
    ----------
    time_series : pandas dataframe
        Dataframe containing the time series data.

    grid_id : str
        Grid ID from which to extract the mean and median value.

    statistic_type : {'mean', 'median'}
        Statistic to extract.

    Returns
    -------
    statistic : float
        Calculated statistic.

    Example
    -------
    >>>
    >>>
    >>>
    >>>
    """
    # Extract statistic
    if statistic_type == "mean":
        statistic = time_series[[grid_id]].mean()[0]
    elif statistic_type == "median":
        statistic = time_series[[grid_id]].median()[0]
    else:
        raise ValueError("Invalid statistic. Must be 'mean' or 'median'.")

    return statistic


def convert_deltas_to_arrays(time_delta, no2_delta, time_series, grid_id):
    """Converts timestamp and NO2 deltas to arrays, for plotting.

    Parameters
    ----------
    time_delta : list
        List containing the timestamp delta values.

    no2_delta : list
        List containing the NO2 delta values.

    time_series : pandas dataframe
        Dataframe containing the time series data.

    grid_id : str
        Grid ID from which to extract the mean and median value.

    Returns
    -------
    tuple

        time_delta_arr : numpy array
            Array containing the time delta values, converted to hours.

        no2_delta_arr : numpy array
            Array containing the NO2 delta values.

        no2_delta_standardized_mean_arr : numpy array
            Array containing the NO2 delta values, as a percent of the
            grid mean value.

        no2_delta_standardized_median_arr : numpy array
            Array containing the NO2 delta values, as a percent of the
            grid median value.

    Example
    -------
    >>>
    >>>
    >>>
    >>>
    """
    # Convert deltas to arrays
    time_delta_arr = format_time_delta(time_delta=time_delta)
    no2_delta_arr = format_no2_delta(no2_delta=no2_delta)
    no2_delta_standardized_mean_arr = standardize_no2_delta(
        no2_delta=no2_delta,
        standard_metric=time_series[[grid_id]].mean()[0],
    )
    no2_delta_standardized_median_arr = standardize_no2_delta(
        no2_delta=no2_delta,
        standard_metric=time_series[[grid_id]].median()[0],
    )

    return (
        time_delta_arr,
        no2_delta_arr,
        no2_delta_standardized_mean_arr,
        no2_delta_standardized_median_arr,
    )


def standardize_no2_delta(no2_delta, standard_metric):
    """Calculates the NO2 delta as a percent of the specified metric.

    Parameters
    ----------
    no2_delta : list
        List containing the NO2 delta values.

    Returns
    -------
    standardized : numpy array
        Array containing the NO2 delta values, as a percent of the
        specified metric (mean, median, max, etc.).

    Example
    -------
    >>>
    >>>
    >>>
    >>>
    """
    # Standardize deltas
    standardized = np.array(
        [delta / standard_metric * 100 for delta in no2_delta]
    )

    return standardized


def plot_deltas(
    time,
    no2,
    grid_id,
    x_max=30,
    y_max=None,
    x_label="Time Difference (hours)",
    y_label=r"NO2 ($\mathrm{mol \cdot m^{-2}}$)",
    standard_metric_value=None,
    standard_metric_title="Mean",
    data_location="South Korea",
    data_source="European Space Agency",
):
    """Plots the NO2 deltas vs. the timestamp deltas for a grid cell.

    Parameters
    ----------
    time : numpy array
        Array containing the time delta values, in hours.

    no2 : numpy array
        Array containing the NO2 delta values, raw or standardized.

    x_max : int or float, optional
        Maximum limit (in hours) for the x-axis. Default value is 30.

    y_max : int or float, optional
        Maximum limit for the y-axis. Default value is None.

    x_label : str, optional
        Label for the x-axis. Default value is 'Time Difference (hours)'.

    y_label : str, optional
        Label for the y-axis.

    data_location : str, optional
        Location of the data. Default value is 'South Korea'.

    data_source : str, optional
        Location of the data. Default value is 'European Space Agency'.

    Returns
    -------
    tuple

        fig : matplotlib.figure.Figure object
            The figure object associated with the plot.

        ax : matplotlib.axes._subplots.AxesSubplot object
            The axes object associated with the plot.

    Example
    ------
        >>>
        >>>
        >>>
        >>>
    """
    # Plot NO2 delta vs. timestamp delta
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.scatter(
            time,
            no2,
            marker="o",
            s=25,
            color="#ff7f00",
            label="NO2 Delta",
        )

        # Add standard metric
        if standard_metric_value:
            ax.axhline(
                standard_metric_value,
                color="#e41a1c",
                label=f"{standard_metric_title}",
                linewidth=2,
                zorder=1,
            )

        # Configure axes, legend, caption
        ax.set_title(
            f"NO2 Deltas, {data_location}, Grid {grid_id}",
            fontsize=24,
        )
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        ax.set_xlim(0, x_max)
        if y_max:
            ax.set_ylim(0, y_max)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        ax.legend(shadow=True, edgecolor="white", fontsize=20, loc="best")
        fig.text(
            0.5,
            0.02,
            f"Data Source: {data_source}",
            ha="center",
            fontsize=16,
        )

    return fig, ax
