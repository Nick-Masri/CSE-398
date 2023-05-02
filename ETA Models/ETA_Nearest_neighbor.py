from rtree import index
import pandas as pd
import ast

# R-Tree index
idx = index.Index()



for i, row in road_segments.iterrows():
    start_coords = row['start_coords']
    end_coords = row['end_coords']
    y1, x1 = start_coords
    y2, x2 = end_coords
    idx.insert(i, (x1, x1, y1, y1), None)
    idx.insert(i, (x2, x2, y2, y2), None)



# get gps point

