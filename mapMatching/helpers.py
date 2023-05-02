from geopy.distance import great_circle
from math import pi

def get_closest_point(point, road):
    gps_lon, gps_lat = point['longitude'], point['latitude']
    start_lat, start_lon = road['start_coords']
    end_lat, end_lon = road['end_coords']

    # Convert coordinates to radians
    lat1, lon1 = gps_lat * pi / 180, gps_lon * pi / 180
    lat2, lon2 = start_lat * pi / 180, start_lon * pi / 180
    lat3, lon3 = end_lat * pi / 180, end_lon * pi / 180

    # Calculate Haversine distance between GPS point and each endpoint
    distance1 = great_circle((gps_lat, gps_lon), (start_lat, start_lon)).kilometers
    distance2 = great_circle((gps_lat, gps_lon), (end_lat, end_lon)).kilometers

    # Find the closest point online segment to GPS point
    dx = lat3 - lat2
    dy = lon3 - lon2
    if dx == 0 and dy == 0:
        t = 0
    else:
        t = ((lat1 - lat2) * dx + (lon1 - lon2) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    closest_lat = lat2 + t * dx
    closest_lon = lon2 + t * dy

    # Convert the closest point back to degrees
    closest_lat_degrees = closest_lat * 180 / pi
    closest_lon_degrees = closest_lon * 180 / pi

    return closest_lat_degrees, closest_lon_degrees

def get_distance(gps_point, road_point):
    gps_lon, gps_lat = gps_point['longitude'], gps_point['latitude']
    lat, lon = road_point[0], road_point[1]
    return great_circle((gps_lat, gps_lon), (lat, lon)).kilometers
