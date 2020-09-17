""" Module containing functions that require the ArcPy site package """

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
    # Extract spatial reference
    spatial_reference = arcpy.Describe(shapefile_path).spatialReference

    return (spatial_reference.name, spatial_reference.linearUnitName)
