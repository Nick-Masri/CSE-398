import math
import pandas as pd
import numpy as np
from geopy.distance import great_circle
from shapely.geometry import Point, box, Polygon
from scipy.stats import norm
from math import radians, sin, cos, sqrt, atan2, exp, pi
import os

""" Methods for use in map matching algorithm """


def get_closest_point(point, road):
    gps_lon, gps_lat = point['longitude'], point['latitude']
    start_lat, start_lon = road['start']
    end_lat, end_lon = road['end']

    # Convert coordinates to radians
    lat1, lon1 = gps_lat * pi / 180, gps_lon * pi / 180
    lat2, lon2 = start_lat * pi / 180, start_lon * pi / 180
    lat3, lon3 = end_lat * pi / 180, end_lon * pi / 180

    # Calculate Haversine distance between GPS point and each endpoint
    distance1 = great_circle((gps_lat, gps_lon), (start_lat, start_lon)).kilometers
    distance2 = great_circle((gps_lat, gps_lon), (end_lat, end_lon)).kilometers

    # Find closest point on line segment to GPS point
    dx = lat3 - lat2
    dy = lon3 - lon2
    t = ((lat1 - lat2) * dx + (lon1 - lon2) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    closest_lat = lat2 + t * dx
    closest_lon = lon2 + t * dy

    # Convert closest point back to degrees
    closest_lat_degrees = closest_lat * 180 / pi
    closest_lon_degrees = closest_lon * 180 / pi

    return (closest_lat_degrees, closest_lon_degrees)


def get_distance(gps_point, road_point):
    gps_lon, gps_lat = gps_point['longitude'], gps_point['latitude']
    lat, lon = road_point[0], road_point[1]
    return great_circle((gps_lat, gps_lon), (lat, lon)).kilometers


class Match_GPS_Trajectory():
    def __init__(self, network, gps_data, dist_tolerance=5, scale=4, beta=3):

        # params
        self.dist_tolerance = dist_tolerance
        self.scale = scale
        self.beta = beta

        # load gps gps_data
        print(f"Size of gps_data: {gps_data.shape}")
        self.gps_data = gps_data

        # load network
        print(f'Size of Network {network.shape}')
        self.network = network

        ### define matrices
        # matrix containing the closest point on a road network to a gps point
        self.cp_mat = np.zeros((len(self.gps_data), len(self.network), 2), dtype=np.float64)
        # matrix containing the distance between closest point and gps point
        self.cd_mat = np.zeros((len(self.gps_data), len(self.network)), dtype=np.float64)

        # calculate observation probabilities
        print("\n\nCalculating Observation Probability")
        print(f"Number of iterations (network * gps data size) = {len(self.gps_data) * len(self.network)}")
        self.obs_prob = self.calc_obs_prob()

        # Calcualte the transition probabilities
        print("\n\nCalculating Transition Probability")
        print("Total Num. Iterations for solution (Road Net. size^2 * GPS data size): {size}".format(
            size=len(self.gps_data) * len(self.network) ** 2))
        self.trans_prob = self.calc_trans_prob()

        # Calculating State Sequences
        print("\n\nApplying Viterbi Algorithm")
        self.state_seq = self.viterbi_algorithm()
        self.road_edge_ids = self.network.iloc[self.state_seq.tolist()].index
        print("Finished")

    def calc_obs_prob(self):
        """
        Calculates the observation probabilities using the HMM algorithm.

        Returns:
        - obs_prob: a 2D numpy array of observation probabilities
        """
        obs_prob = np.zeros((len(self.gps_data), len(self.network)), dtype=np.float64)

        T = len(self.gps_data)
        R = len(self.network)

        for t in range(T):
            for r in range(R):

                gps_point = self.gps_data.iloc[t]
                road_info = self.network.iloc[r]

                closest_point = get_closest_point(gps_point, road_info)
                distance = get_distance(gps_point, closest_point)

                # fill matrices
                self.cp_mat[t, r, 0] = closest_point[0]
                self.cp_mat[t, r, 1] = closest_point[1]
                self.cd_mat[t, r] = distance

                if distance > self.dist_tolerance:
                    obs_prob[t, r] = 0
                else:
                    prob = norm.pdf(distance, loc=0, scale=self.scale)
                    if np.isnan(prob):
                        # set probability to 1.0 when distance is 0
                        obs_prob[t, r] = 1.0
                    else:
                        obs_prob[t, r] = prob

        return obs_prob

    def calc_trans_prob(self):
        """
        Calculates the transition probabilities using the HMM algorithm.

        Returns:
        - trans_prob: a 3D numpy array of transition probabilities
        """
        T = len(self.gps_data)
        R = len(self.network)
        trans_mat = np.zeros((T - 1, R, R))

        # probability we transition from r_i to r_j
        for t in range(T - 1):
            for r_i in range(R):
                # if the probablity of being on road r_i at time t is zero, we don't have to calculate the trans prob
                if self.obs_prob[t, r_i] != 0:
                    for r_j in range(R):
                        # if the probablity of being on road r_j  at time t+1 is zero, we don't have to calculate the trans prob
                        if self.obs_prob[t + 1, r_j] != 0:

                            # get gps points
                            z_t = self.gps_data.iloc[t]
                            z_t1 = self.gps_data.iloc[t + 1]

                            if r_i == r_j:
                                d_t = get_distance(z_t, z_t1)
                            else:

                                # should be graph route distance, descent proxy for now
                                route_dist = abs(self.cp_mat[t, r_i, 0] - self.cp_mat[t, r_j, 0])
                                + abs(self.cp_mat[t, r_i, 1] - self.cp_mat[t, r_j, 1])
                                d_t = get_distance(z_t, z_t1) - route_dist

                            trans_mat[t, r_i, r_j] = (1 / self.beta) * np.exp(-d_t / self.beta)

        return trans_mat

    def viterbi_algorithm(self):
        """
        Given a sequence of observations, the start probabilities, transition probabilities, and
        emission probabilities, computes the most likely sequence of hidden states (i.e., the state
        sequence that maximizes the joint probability of the observed sequence and the state sequence).

        Args:
        - trans_p (array-like of shape (n_observations - 1, n_states, n_states)): The probabilities of
          transitioning between states at each time step
        - obs_p (array-like of shape (n_observations, n_states)): The probabilities of each observation
          being obsted from each state
        """

        # Determine the number of observations and states
        n_obs = self.obs_prob.shape[0]
        n_states = self.obs_prob.shape[1]

        # Initialize the viterbi and backpointer tables
        viterbi_table = np.zeros((n_obs, n_states))
        backpointer_table = np.zeros((n_obs, n_states), dtype=int)

        # Initialize the first row of the viterbi table with the start probabilities
        viterbi_table[0, :] = self.obs_prob[0, :]

        # Loop through the remaining observations
        for t in range(1, n_obs):
            # Loop through the states
            for r in range(n_states):
                # Calculate the scores for transitioning to this state from each previous state
                trans_scores = viterbi_table[t - 1, :] * self.trans_prob[t - 1, :, r]

                # Calculate the maximum score and corresponding backpointer
                max_score = np.max(trans_scores) * self.obs_prob[t, r]
                backpointer = np.argmax(trans_scores)

                # Update the viterbi and backpointer tables with the new values
                viterbi_table[t, r] = max_score
                backpointer_table[t, r] = backpointer

        # Determine the sequence of states with the highest probability
        state_seq = np.zeros(n_obs, dtype=int)
        state_seq[-1] = np.argmax(viterbi_table[-1, :])
        for t in range(n_obs - 2, -1, -1):
            state_seq[t] = backpointer_table[t + 1, state_seq[t + 1]]

        return state_seq
