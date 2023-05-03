import os
import time
import pandas as pd
import ast
from mapMatching.mapMatchFaster import MapMatch
import numpy as np

if __name__ == '__main__':
    # Check if train.csv and trips.csv exist
    if os.path.exists('../outputs/train.csv') and os.path.exists('../outputs/trips.csv'):
        print('train.csv and trips.csv already exist. Skipping data generation...')
    else:
        # Run the data generation script
        print('Generating training and trips data...')
        os.system('python3 generate/genData.py')

    # # check if network exists
    # if os.path.exists('outputs/osmnx-df.csv'):
    #     print('Network already exists. Skipping data generation...')
    # else:
    #     print("Generating Road Network...")
    #     os.system('python3 generate/genMap.py')

    # load trips and train data
    trips = pd.read_csv('../outputs/trips.csv', index_col=0)
    train_data = pd.read_csv('../outputs/train.csv', index_col=0)

    # load the network
    network = pd.read_csv('../outputs/osmnx-df.csv')
    network['start_coords'] = network['start_coords'].apply(ast.literal_eval)
    network['end_coords'] = network['end_coords'].apply(ast.literal_eval)

    # create df structure
    if os.path.exists('../outputs/results.csv'):
        results = pd.read_csv('../outputs/results.csv')
    else:
        results = pd.DataFrame(columns=["cabID", "tripID", "time", "gps_point",
                                               "road_segment", "road_point"])

    for idx, trip in trips.iloc[2945:].iterrows():
        if len(results[(results.tripID == idx) & (results.cabID == trip['cabID'])]) > 0:
            print(f'Result already exists, skipping map matching for trip {idx}')
            continue

        print("\n\n#####################################")
        print(f"# Map Matching for Trip {idx}")
        print("#####################################")
        start_time = time.time()
        data = train_data[train_data.cab_id == trip.cabID]

        # for some reason some of the trips have the start index after the end index
        # need to fix this error in the data generation
        if trip['tripStartIDX'] > trip['tripEndIDX']:
            print("Error: tripStartIDX > tripEndIDX")
            continue

        relevant_gps_data = data.loc[trip['tripStartIDX']:trip['tripEndIDX'] + 1]
        relevant_gps_data = relevant_gps_data[['latitude', 'longitude', 'time']]

        try:
            map_match = MapMatch(network, relevant_gps_data)
        except np.core._exceptions._ArrayMemoryError:
            print(f'Memory Error (network too big) for trip {idx}')
            continue

        closest_points = []
        for road_idx, road in enumerate(map_match.state_seq):
            closest_points.append(map_match.cp_mat[road_idx, road])

        gps_points = [[lat, lon] for lat, lon in zip(relevant_gps_data['latitude'], relevant_gps_data['longitude'])]

        trip_output = pd.DataFrame({
            "cabID": trip['cabID'],
            "tripID": idx,
            "time": relevant_gps_data['time'],
            "gps_point": gps_points,
            "road_segment": map_match.road_edge_ids,
            "road_point": closest_points})

        # Add column for previous road segment
        trip_output["prev_road_segment"] = trip_output["road_segment"].shift(1)
        trip_output["prev_road_point"] = trip_output["road_point"].shift(1)
        trip_output["prev_time"] = trip_output["time"].shift(1)
        results = pd.concat([results, trip_output])

        results.to_csv('outputs/results.csv')

        # calculate total execution time
        total_time = time.time() - start_time
        print(f"Total time: {total_time} seconds")

    print("\n\n#####################################")
    print(f"# Completed Execution of {len(trips)} trips")
    print("#####################################")
