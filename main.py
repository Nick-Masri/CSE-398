import os
import sys

# Check if train.csv and trips.csv exist
if os.path.exists('outputs/train.csv') and os.path.exists('outputs/trips.csv'):
    print('train.csv and trips.csv already exist. Skipping data generation...')
else:
    # Run the data generation script
    print('Generating training and trips data...')
    os.system('python generate/genData.py')

if os.path.exists('outputs/network.pickle'):
    print('Network already exists. Skipping data generation...')
else:
    print("Generating Road Network...")
    os.system('python generate/genMap.py')

# Continue with the rest of the code
# ...
