""" Module containing functions that require the ArcPy site package """

import csv
import arcpy


def get_shapefile_extent(shapefile_path):
    """Extracts and returns the shapefile extent: (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile (.shp).

    Returns
    -------
    extent : tuple
        Tuple containing (xmin, ymin, xmax, ymax).

    Example
    -------
        >>> # Imports
        >>> import arcpy
        >>> # Get spatial extent
        >>> get_shapefile_extent("vermont_boundary.shp")
        (424788.83999999985 25211.83689999953 581554.3701 279798.47360000014)
    """
    # Extract extent
    extent = arcpy.Describe(shapefile_path).extent

    return (extent.XMin, extent.YMin, extent.XMax, extent.YMax)


def get_shapefile_crs(shapefile_path):
    """Extracts and returns the shapefile crs: (name, units).

    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile (.shp).

    Returns
    -------
    extent : tuple
        Tuple containing (name, units).

    Example
    -------
        >>> # Imports
        >>> import arcpy
        >>> # Get CRS info
        >>> get_shapefile_crs("vermont_boundary.shp")
        ('NAD83_Vermont', 'Meter')
    """
    # Get spatial reference object
    spatial_reference = arcpy.Describe(shapefile_path).spatialReference

    # Extract CRS name and units
    crs_name = spatial_reference.name
    crs_units = (
        spatial_reference.linearUnitName
        if spatial_reference.linearUnitName
        else "Degrees"
    )

    return (crs_name, crs_units)


def get_country_name(shapefile):
    """Extracts the country name from a shapefile.

    Expects the file name to match "gadm36_NAME_NAME.shp".

    Parameters
    ----------
    shapefile : str
        File to extract the country name from.

    Returns
    -------
    country_name : str
        Name of the country, extracted from the original shapefile name.

    Example
    -------
        >>> # Get country name
        >>> get_country_name("gadm36_south_korea.shp")
        'South Korea'
    """
    # Extract country name
    country_name = " ".join(
        [word.capitalize() for word in shapefile[7:].split("_")]
    )

    return country_name


# Create list of shapefile names (without .shp if run in ArcGIS project)
country_shapefiles = ["gadm36_singapore", "gadm36_south_korea"]

# Create list of names and exents (for saving to CSV)
country_extents = [
    [get_country_name(shapefile), get_shapefile_extent(shapefile)]
    for shapefile in country_shapefiles
]

# Write country extents to CSV file
with open("country-extents.csv", mode="w", newline="") as csv_file:
    csv_writer = csv.writer(
        csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
    )
    csv_writer.writerow(["Country", "Extent"])
    csv_writer.writerows(country_extents)

# Read CSV to check if the data are in order
with open("country-extents.csv", newline="") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",", quotechar='"')
    for row in csv_reader:
        print(", ".join(row))
