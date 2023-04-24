import pandas as pd
from mapMatching.mapMatch import MapMatch

trips = pd.read_csv('outputs/trips.csv', index_col=0)
train_data = pd.read_csv('outputs/train.csv', index_col=0)
network = pd.read_csv('outputs/osmnx-df.csv')

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

    map_match = MapMatch(network, relevant_gps_data)

all_taxi_locations.to_csv('outputs/mapMatchTest.csv')