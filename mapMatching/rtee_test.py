import ast

from geopy.distance import great_circle
from rtree import index
import pandas as pd
from helpers import get_closest_point, get_distance

# R-Tree index
idx = index.Index()

# calculate bbox
north, south, east, west = 37.9549, 37.3249, -121.9694, -122.6194
width = abs(east - west)
height = abs(north - south)
left = (west - min(east, west)) / width
bottom = (south - min(north, south)) / height
right = (east - min(east, west)) / width
top = (north - min(north, south)) / height

# insert bbox into rtree
bbox = (left, right, bottom, top)
# bbox = (east, west, south, north)
idx.insert(0, bbox, None)

# get some road segments
network = pd.read_csv('../outputs/osmnx-df.csv')
road_segments = network.iloc[0:4].copy()
road_segments['start_coords'] = road_segments['start_coords'].apply(ast.literal_eval)
road_segments['end_coords'] = road_segments['end_coords'].apply(ast.literal_eval)


# get gps point
points = pd.read_csv('../outputs/train.csv')
point = points.iloc[0]

# insert road segments into rtree
for i, row in road_segments.iterrows():
    start_coords = row['start_coords']
    end_coords = row['end_coords']
    y1, x1 = start_coords
    y2, x2 = end_coords
    x1, x2 = min(x1, x2), max(x1,x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    idx.insert(i, (x1, x2, y1, y2), None)


# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return great_circle(point1, point2).miles


# Find the closest road segments
closest_segments = []
point_cord = (point['longitude'], point['latitude'])
for segment_id in idx.nearest(point_cord, 2):
    cp = get_closest_point(point, road_segments.iloc[segment_id])
    distance = get_distance(point, cp)
    closest_segments.append((segment_id, distance))

# Sort and return closest road segments
closest_segments.sort(key=lambda x: x[1])
closest_segments = closest_segments[:2]
print(closest_segments)


# Checking all segments
closest_segments = []
for segment_id in range(4):
    cp = get_closest_point(point, road_segments.iloc[segment_id])
    distance = get_distance(point, cp)
    closest_segments.append((segment_id, distance))

# Sort and return closest road segments
closest_segments.sort(key=lambda x: x[1])
# closest_segments = closest_segments[:2]
print(closest_segments)