import os
import pandas as pd
#
# training_instances = os.listdir('../stage3/train')
#
#
# # Training Data
# training_instances = os.listdir('../training')
#
# # Create an empty list to store the dataframes
# train_data = []
# idx = 0
#
# for instance in training_instances:
#     data = pd.read_csv(f"../training/{instance}", sep=" ", header=None)
#     data.columns = ['latitude', 'longitude', 'occupancy', 'time']
#
#     # Convert time to datetime
#     data['time'] = pd.to_datetime(data['time'], unit='s')
#
#     # sort values by time
#     data.sort_values('time', inplace=True)
#     data = data.reset_index(drop=True)
#
#     # Add the cab ID column
#     data['cab_id'] = idx
#
#     # Increment the cab ID
#     idx += 1
#
#     # Append the dataframe to the list
#     train_data.append(data)
#
# # Concatenate the dataframes into a single dataframe
# train_data = pd.concat(train_data, ignore_index=True)
# train_data.to_csv('../stage3/outputs/train.csv')
# print('train.csv created successfully.')

# Create Trips dataframe
train_data = pd.read_csv('../outputs/train.csv')

trip_list = []
for dataset in train_data.groupby('cab_id'):
    # Get the index to the cabID
    idx = dataset[0]

    # find the indices where the occupancy changes from 0 to 1 or 1 to 0
    occupancy_diff = dataset[1]['occupancy'].diff()

    trip_start_mask = (dataset[1]['occupancy'] == 1) & (occupancy_diff == 1)
    trip_end_mask = (dataset[1]['occupancy'] == 0) & (occupancy_diff == -1)

    trip_starts = dataset[1][trip_start_mask].index
    trip_ends = dataset[1][trip_end_mask].index - 1

    for start, end in zip(trip_starts, trip_ends):
        trip = dataset[1][start:end+1]
        if (len(trip) > 0) and (end > start):
            trip_list.append({'cabID': idx, 'tripStartIDX': start, 'tripEndIDX': end})

trips = pd.DataFrame(trip_list)

# remove rows where the start idx is the same as the end idx
trips = trips[trips['tripStartIDX'] != trips['tripEndIDX']]
trips = trips.reset_index(drop=True)

trips.to_csv('../stage3/outputs/trips.csv', index=False)
print('trips.csv created successfully.')
