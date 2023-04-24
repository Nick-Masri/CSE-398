import os
import osmnx as ox
import pickle
import pandas as pd
from mapDataframe import reformat_df
# from cleanedMapDF import reformat_df

# setup directory

# Get the directory path of the current file
dir_path = os.path.dirname(os.path.abspath(__file__))


# bbox coordinates
north, south, east, west = 37.9549, 37.3249, -121.9694, -122.6194


# Check if the graph file already exists
if os.path.isfile(os.path.join(dir_path, '../outputs/network.pickle')):
    print("OSMNX network already exists, loading from file...")
    with open(os.path.join(dir_path, '../outputs/network.pickle'), 'rb') as f:
        G_simplified = pickle.load(f)
else:
    # Create the graph
    G = ox.graph_from_bbox(north, south, east, west, network_type='drive', simplify=True, retain_all=False,
                           truncate_by_edge=False, clean_periphery=True)

    # Define the projected CRS to use
    crs_proj = '+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'

    # Reproject the graph to the projected CRS
    G = ox.project_graph(G, to_crs=crs_proj)
    print("Size before simplification", G.size(weight='length'))

    # Consolidate graph with a tolerance of 5 meters (it is double the param)
    G_simplified = ox.simplification.consolidate_intersections(G, tolerance=2.5, rebuild_graph=True)
    print("Size after simplification", G_simplified.size(weight='length'))

    # Calculate the size reduction percentage
    size_reduction_pct = (1 - G_simplified.size(weight='length') / G.size(weight='length')) * 100

    # Print the size reduction percentage
    print(f"Graph size reduced by {size_reduction_pct:.2f}%")

    # Save G_simplified to a file
    with open(os.path.join(dir_path, '../outputs/network.pickle'), 'wb') as f:
        pickle.dump(G_simplified, f)

    print("OSMNX Network Created and saved.")

# Check if the dataframe file already exists
if os.path.isfile(os.path.join(dir_path, '../outputs/network.csv')):
    print("Dataframe already exists, loading from file...")
    df = pd.read_csv(os.path.join(dir_path, '../outputs/network.csv'))
else:
    # Convert the graph to a dataframe
    df = ox.graph_to_gdfs(G_simplified, nodes=False, edges=True)

    # reformat the df
    df = reformat_df(df)

    # Save the dataframe to a file
    df.to_csv(os.path.join(dir_path, '../outputs/network.csv'), index=False)

    print("Dataframe created and saved.")

# Do something with the dataframe
# ...
