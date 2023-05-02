import ast
import taxicab as tc
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import datetime
import osmnx as ox

df = pd.read_csv('../stage3/test_student.txt', sep=' ', header=None, index_col=0)
df.columns = ['file', 'start_lat', 'start_long', ' time', 'end_lat', 'end_long']
df['time'] = pd.to_datetime(df[' time'], unit='s')

with open('../outputs/network.pickle', 'rb') as f:
    G = pickle.load(f)

# impute missing edge speeds and add travel times
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

warnings.simplefilter(action='ignore', category=FutureWarning)

eta_results = pd.DataFrame(columns=['index', 'distance', 'route', 'start_time'])

for idx, row in df.iterrows():
    # calculate distance
    try:
        path = tc.distance.shortest_path(G, (row['start_lat'], row['start_long']), (row['end_lat'], row['end_long']))
        dist = path[0]
        route = path[1]
        print(f"Row {idx} completed")
        eta_results = eta_results.append({'index': idx, 'distance': dist, 'route': route, 'start_time': row['time']},
                                         ignore_index=True)
    except Exception as e:
        print(f"Error occurred for row {idx}: {e}")
        eta_results = eta_results.append({'index': idx, 'distance': np.nan, 'route': np.nan, 'start_time': row['time']},
                                         ignore_index=True)

eta_results.to_csv('outputs/eta_dist.csv', index=False)


eta_results['eta_time'] = np.nan

for idx, row in eta_results.iterrows():
    route_nodes = row['route']
    travel_time = sum(ox.utils_graph.get_route_edge_attributes(G, route_nodes, attribute='travel_time'))
    eta_time = pd.to_datetime(row['start_time']) + datetime.timedelta(seconds=travel_time)
    eta_results.at[idx, 'eta_time'] = eta_time

final_results = eta_results[['index', 'eta_time']]
final_results.columns = ['id', 'eta']
final_results.to_csv('outputs/eta_distance.csv', index=False)
