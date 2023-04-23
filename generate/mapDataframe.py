import pandas as pd
import ast
import shapely.wkt


def filter_df(df):
    # Create a Boolean mask indicating which rows have end coordinates with latitudes within the valid range.
    lat_range_filter = (df['end_coords'].apply(lambda x: -90 <= x[0] <= 90))

    # Apply the mask to the DataFrame and return the filtered DataFrame.
    edges_coords_filtered = df[lat_range_filter]

    return edges_coords_filtered


def one_2_one_osmid_lanes(df):
    # create a new dataframe with one-to-one mapping between osmid and lanes
    df['osmid'] = df['osmid'].apply(lambda x: x if isinstance(x, list) else [x])
    df['lanes'] = df['lanes'].apply(lambda x: x if isinstance(x, list) else [x])

    osmids = []
    lanes = []

    for osmid, lane in zip(df['osmid'], df['lanes']):
        for i, os in enumerate(osmid):
            if isinstance(lane, list) and i < len(lane):
                lanes.append(lane[i])
            else:
                lanes.append(lane)
            osmids.append(os)

    new_df = pd.DataFrame({'osmid': osmids, 'lanes': lanes})
    return new_df


# Function to reformat a given DataFrame
def reformat_df(df):
    # List of column names to keep
    cols_to_keep = ['osmid', 'oneway', 'lanes', 'ref', 'name', 'highway', 'maxspeed', 'reversed', 'length', 'geometry']

    # Select only the columns in cols_to_keep and assign it back to the same variable
    df = df[cols_to_keep]

    # Reset the index of the DataFrame
    df = df.reset_index(drop=True)

    # Rename certain columns in the DataFrame
    df = df.rename(columns={'u': 'start_node', 'v': 'end_node', 'name': 'road_name'})

    # Reorder the columns in the DataFrame
    df = df[
        ['osmid', 'start_node', 'end_node', 'oneway', 'lanes', 'ref', 'road_name', 'highway', 'maxspeed', 'reversed',
         'length', 'geometry']]

    # Generate a DataFrame that maps each OSM ID to a lane count
    osmid_lanes_df = one_2_one_osmid_lanes(df)

    # Concatenate the original DataFrame and the osmid_lanes_df on axis 1
    df = pd.concat([df, osmid_lanes_df], axis=1)

    # Drop rows where either the 'osmid' or 'maxspeed' column has a missing value
    df = df.dropna(subset=['osmid', 'maxspeed'])

    # Select all columns except the first one
    df = df.iloc[:, 1:]

    # Explode the 'lanes' column so that each lane count value gets its own row
    df = df.explode('lanes')

    # Get the start and end coordinates for each row in the DataFrame
    df_w_coords = get_start_end_points(df)

    # Concatenate the df_w_coords DataFrame and the original DataFrame on axis 1
    df = pd.concat([df_w_coords, df], axis=1)

    # Create a new column 'edge_id' that concatenates the start_node and end_node columns
    df['edge_id'] = df['start_node'].astype(str) + '-' + df['end_node'].astype(str)

    # Set the 'edge_id' column as the index of the DataFrame
    df.set_index('edge_id', inplace=True)

    # Filter the DataFrame to remove certain rows
    df = filter_df(df)

    # Reconfigure and filter the DataFrame
    df = reconfigure_n_filter(df)

    return df


def reconfigure_n_filter(df):
    # create a copy of the DataFrame
    df = df.copy()

    # convert string representations of lists to actual lists of coordinates
    df['start_coords'] = df['start_coords'].apply(ast.literal_eval)
    df['end_coords'] = df['end_coords'].apply(ast.literal_eval)

    # convert Well-Known Text (WKT) geometries to Shapely geometries
    df['geometry'] = df['geometry'].apply(shapely.wkt.loads)

    # remove duplicates from the DataFrame
    df.drop_duplicates(inplace=True)

    # create separate DataFrames for start and end coordinates
    start_coords = pd.DataFrame(df['start_coords'].tolist(), columns=['start_lat', 'start_lon'])
    end_coords = pd.DataFrame(df['end_coords'].tolist(), columns=['end_lat', 'end_lon'])

    # remove rows where start and end coordinates are identical
    df = df.loc[
        ~((start_coords['start_lat'] == start_coords['start_lon']) | (end_coords['end_lat'] == end_coords['end_lon']))]

    # filter out rows where length is less than or equal to 125
    df_filtered = df[df['length'] > 125]

    # reset index and return filtered DataFrame
    df = df_filtered.reset_index(drop=True)
    return df


def get_start_end_points(df):
    # create a DataFrame to store start and end coordinates
    coords_df = pd.DataFrame(columns=['start_coords', 'end_coords'])

    # loop through each row in the DataFrame
    for idx, row in df.iterrows():
        # extract the geometry (a LineString object) from the 'geometry' column
        linestring = row['geometry']

        # get the start and end points of the LineString
        start_point = linestring.coords[0]
        end_point = linestring.coords[-1]

        # store the start and end coordinates in a new row in the DataFrame
        coords_df.loc[idx] = [(start_point[1], start_point[0]), (end_point[1], end_point[0])]

    return coords_df
