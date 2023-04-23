import folium
import pandas as pd

# Load the CSV file
df = pd.read_csv('../outputs/train.csv')

# Define the bounding box
north, south, east, west = 37.9549, 37.3249, -121.9694, -122.6194

# Create the Folium map centered on the bounding box
map_center = [(north + south) / 2, (east + west) / 2]
m = folium.Map(location=map_center, zoom_start=12)

# Add the bounding box to the map as a rectangle
bbox = [[south, west], [north, east]]
folium.Rectangle(
    bounds=bbox,
    fill=False,
    color='red',
    weight=2,
    opacity=1
).add_to(m)

# Add markers for minimum and maximum latitude and longitude points
min_lat, min_lat_long = df.loc[df['latitude'].idxmin()][['latitude', 'longitude']]
max_lat, max_lat_long = df.loc[df['latitude'].idxmax()][['latitude', 'longitude']]
folium.Marker(location=[min_lat, min_lat_long]).add_to(m)
folium.Marker(location=[max_lat, max_lat_long]).add_to(m)

min_long_lat, min_long = df.loc[df['longitude'].idxmin()][['latitude', 'longitude']]
max_long_lat, max_long = df.loc[df['longitude'].idxmax()][['latitude', 'longitude']]
folium.Marker(location=[min_long_lat, min_long]).add_to(m)
folium.Marker(location=[max_long_lat, max_long]).add_to(m)

# Display the map
m.save("bbox.html")
