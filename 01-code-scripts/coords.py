""" Define string for polygon bounding box """

# Bounding box for Centre County, PA
lon_min = -78.4
lon_max = -77.1
lat_min = 40.6
lat_max = 41.3

# Create polygon string
polygon = (
    "POLYGON(("
    f"{lon_min} {lat_min},"  # Bottom-Left
    f"{lon_min} {lat_max},"  # Top-Left
    f"{lon_max} {lat_max},"  # Top-Right
    f"{lon_max} {lat_min},"  # Bottom-Right
    f"{lon_min} {lat_min}))"  # Bottom-Left
)

print(polygon)
