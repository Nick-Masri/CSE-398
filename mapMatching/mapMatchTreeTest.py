import pandas as pd
from mapMatching.mapMatchTree import MapMatch
import ast
from rtree import index
import time
import pickle
import numpy as np

start_time = time.time()

# load network
network = pd.read_csv('../outputs/osmnx-df.csv')
network['start_coords'] = network['start_coords'].apply(ast.literal_eval)
network['end_coords'] = network['end_coords'].apply(ast.literal_eval)

# load the rtree
rtree = index.Index('outputs/rtree')

# load trips
trips = pd.read_csv('../outputs/trips.csv', index_col=0)
train_data = pd.read_csv('../outputs/train.csv', index_col=0)

# create df structure
all_taxi_locations = pd.DataFrame(columns=["cabID", "tripID", "time", "gps_point",
                                           "road_segment", "road_point"])

print(network.columns)
for idx, trip in trips.iloc[0:1].iterrows():
    print("\n\n#####################################")
    print(f"# Map Matching for Trip {idx}")
    print("#####################################")

    data = train_data[train_data.cab_id == trip.cabID]

    relevant_gps_data = data.iloc[trip['tripStartIDX']:trip['tripEndIDX'] + 1]
    relevant_gps_data = relevant_gps_data[['latitude', 'longitude', 'time']]

    map_match = MapMatch(network, relevant_gps_data, rtree, k=50)

# save df
all_taxi_locations.to_csv('outputs/mapMatchTest.csv')

# print(map_match.obs_prob)
# print(map_match.obs_prob[np.nonzero(map_match.obs_prob)])
# print(map_match.trans_prob[0][np.nonzero(map_match.trans_prob[0])])
print(map_match.state_seq)
# print(map_match.trans_prob[0])
# print(map_match.close_roads[0])


# calculate total execution time
total_time = time.time() - start_time
print(f"Total time: {total_time} seconds")

nonzero_entries = []  # to store the (row, column) tuples of nonzero entries
matrix = map_match.obs_prob
T = len(relevant_gps_data)
R = len(network)
row, col = matrix.nonzero()
route_indices = [j for i, j in zip(*matrix.nonzero())]
print(route_indices)
