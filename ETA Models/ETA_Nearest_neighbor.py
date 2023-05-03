from rtree import index
import pandas as pd
import ast
import os

# R-Tree index
idx = index.Index()

taxi_train_etas = os.listdir('../stage3/outputs/trips_by_taxi')

class ETA:
    def __init__(self, start_time=None, end_time=None):
        self.start_time = start_time
        self.end_time = end_time


for taxi in taxi_train_etas:
    data = pd.read_csv(f"../stage3/outputs/trips_by_taxi/{taxi}")
    for i, row in data.iterrows():
        sTime = row['source_timestamp']
        dTime = row['destination_timestamp']

        start_eta = ETA(start_time=sTime)
        end_eta = ETA(end_time=dTime)
        start_point = row['origin_point']
        end_point = row['destination_point']

        x1, y1 = ast.literal_eval(start_point)[1], ast.literal_eval(start_point)[0]
        x2, y2 = ast.literal_eval(end_point)[1], ast.literal_eval(end_point)[0]

        idx.insert(i, (x1, y1), start_eta)
        idx.insert(i, (x2, y2), end_eta)

test_data = pd.read_csv('../stage3/test_student.txt', sep=' ', header=None, index_col=0)

test_point = test_data.iloc[0]
print(test_point)
# print(idx.nearest())


