import os
import pandas as pd

# Create the outputs directory if it doesn't exist
if not os.path.exists('../outputs'):
    os.makedirs('../outputs')

# bbox coordinates
north, south, east, west = 37.9549, 37.3249, -121.9694, -122.6194

# Check if train.csv exists
if os.path.exists('../outputs/train.csv'):
    print('train.csv already exists. Skipping creation...')
else:
    # Training Data
    training_instances = os.listdir('../training')
    # Create an empty list to store the dataframes
    train_data = []

    for idx, instance in enumerate(training_instances):
        data = pd.read_csv(f"training/{instance}", sep=" ", header=None)
        data.columns = ['latitude', 'longitude', 'occupancy', 'time']

        # Convert time to datetime
        data['time'] = pd.to_datetime(data['time'], unit='s')

        # sort values by time
        data.sort_values('time', inplace=True)
        data = data.reset_index(drop=True)

        # Check if any coordinate is outside the bbox
        mask = (data['latitude'] < south) | (data['latitude'] > north) | \
               (data['longitude'] < west) | (data['longitude'] > east)
        if mask.any():
            continue  # Skip this instance if it has coordinates outside the bbox

        # Add the cab ID column
        data['cab_id'] = idx

        # Append the dataframe to the list
        train_data.append(data)

    # Concatenate the dataframes into a single dataframe
    train_data = pd.concat(train_data, ignore_index=True)
    train_data.to_csv('outputs/train.csv')
    print('train.csv created successfully.')

# Create Trips dataframe

# Check if trips.csv exists
if os.path.exists('../outputs/trips.csv'):
    print('trips.csv already exists. Skipping creation...')
else:
    # Load the train data
    train_data = pd.read_csv('../outputs/train.csv')

    trip_list = []
    for idx, dataset in enumerate(train_data.groupby('cab_id')):
        # find the indices where the occupancy changes from 0 to 1 or 1 to 0
        occupancy_diff = dataset[1]['occupancy'].diff()

        trip_start_mask = (dataset[1]['occupancy'] == 1) & (occupancy_diff == 1)
        trip_end_mask = (dataset[1]['occupancy'] == 0) & (occupancy_diff == -1)

        trip_starts = dataset[1][trip_start_mask].index
        trip_ends = dataset[1][trip_end_mask].index - 1

        for start, end in zip(trip_starts, trip_ends):
            trip = dataset[1][start:end+1]
            # Check if any point in the trip is outside the bbox
            if not ((trip['latitude'] > north) | (trip['latitude'] < south) |
                    (trip['longitude'] > east) | (trip['longitude'] < west)).any():
                trip_list.append({'cabID': idx, 'tripStartIDX': start, 'tripEndIDX': end})

    trips = pd.DataFrame(trip_list)

    # remove rows where the start idx is the same as the end idx
    trips = trips[trips['tripStartIDX'] != trips['tripEndIDX']]
    trips = trips.reset_index(drop=True)

    trips.to_csv('outputs/trips.csv')
    print('trips.csv created successfully.')
