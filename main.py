import os
import time
import pandas as pd
import ast
from mapMatching.mapMatchFaster import MapMatch

if __name__ == '__main__':
    # Check if train.csv and trips.csv exist
    if os.path.exists('outputs/train.csv') and os.path.exists('outputs/trips.csv'):
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
    trips = pd.read_csv('outputs/trips.csv', index_col=0)
    train_data = pd.read_csv('outputs/train.csv', index_col=0)

    # load the network
    network = pd.read_csv('outputs/osmnx-df.csv')
    network['start_coords'] = network['start_coords'].apply(ast.literal_eval)
    network['end_coords'] = network['end_coords'].apply(ast.literal_eval)

    # create df structure
    all_taxi_locations = pd.DataFrame(columns=["cabID", "tripID", "time", "gps_point",
                                               "road_segment", "road_point"])

    for idx, trip in trips.iterrows():
        print("\n\n#####################################")
        print(f"# Map Matching for Trip {idx}")
        print("#####################################")
        start_time = time.time()
        data = train_data[train_data.cab_id == trip.cabID]

        relevant_gps_data = data.iloc[trip['tripStartIDX']:trip['tripEndIDX'] + 1]
        relevant_gps_data = relevant_gps_data[['latitude', 'longitude', 'time']]

        map_match = MapMatch(network, relevant_gps_data)

        closest_points = []
        for road_idx, road in enumerate(map_match.state_seq):
            closest_points.append(map_match.cp_mat[road_idx, road])

        gps_points = [[lat, lon] for lat, lon in zip(relevant_gps_data['latitude'], relevant_gps_data['longitude'])]

        trip_output = pd.DataFrame({
            "cabID": trip['cabID'] * len(relevant_gps_data),
            "tripID": idx * len(relevant_gps_data),
            "time": relevant_gps_data['time'],
            "gps_point": gps_points,
            "road_segment": map_match.road_edge_ids,
            "road_point": closest_points})

        # Add column for previous road segment
        trip_output["prev_road_segment"] = trip_output["road_segment"].shift(1)
        trip_output["prev_road_point"] = trip_output["road_point"].shift(1)
        trip_output["prev_time"] = trip_output["time"].shift(1)
        all_taxi_locations = pd.concat([all_taxi_locations, trip_output])

        all_taxi_locations.to_csv('outputs/results.csv')

        # calculate total execution time
        total_time = time.time() - start_time
        print(f"Total time: {total_time} seconds")

    print("\n\n#####################################")
    print(f"# Completed Execution of {len(trips)} trips")
    print("#####################################")


