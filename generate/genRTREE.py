import pandas as pd
from rtree import index
import pickle
import ast

# R-Tree index
rtree = index.Index('../outputs/rtree')

network = pd.read_csv('../outputs/osmnx-df.csv')
network['start_coords'] = network['start_coords'].apply(ast.literal_eval)
network['end_coords'] = network['end_coords'].apply(ast.literal_eval)

# insert road segments into rtree
for i, row in network.iterrows():
    start_coords = row['start_coords']
    end_coords = row['end_coords']
    y1, x1 = start_coords
    y2, x2 = end_coords
    x1, x2 = min(x1, x2), max(x1,x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    rtree.insert(i, (x1, x2, y1, y2), None)

print(len(rtree))

rtree.close()
# # serialize the index using pickle
# with open("../outputs/rtree.pickle", "wb") as f:
#     pickle.dump(rtree, f)
