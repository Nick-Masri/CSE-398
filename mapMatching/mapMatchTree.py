import numpy as np
from scipy.stats import norm
import scipy.sparse as sp
from mapMatching.helpers import get_closest_point, get_distance


class MapMatch:
    def __init__(self, network, gps_data, rtree, dist_tolerance=0.5, scale=4, beta=3, k=10):

        # params
        self.dist_tolerance = dist_tolerance
        self.scale = scale
        self.beta = beta
        self.k = k
        self.close_roads = {}
        self.seen = set()

        # load gps gps_data
        print(f"Size of gps_data: {gps_data.shape}")
        self.gps_data = gps_data

        # load network and rtree
        print(f'Size of Network {network.shape}')
        self.network = network
        self.rtree = rtree

        # todo: filter network (rtree) using bbox of coordinates
        # todo: figure out route distance

        # define matrices
        # matrix containing the closest point on a road network to a gps point
        self.cp_mat = np.zeros((len(self.gps_data), len(self.network), 2), dtype=np.float64)
        # matrix containing the distance between the closest point and gps point
        self.cd_mat = np.zeros((len(self.gps_data), len(self.network), 2), dtype=np.float64)

        # calculate observation probabilities
        print("\n\nCalculating Observation Probability")
        print(f"Number of iterations (k * gps data size) = {len(self.gps_data) * self.k}")
        self.obs_prob = self.calc_obs_prob()

        # Calculate the transition probabilities
        print("\n\nCalculating Transition Probability")
        print("Total Num. Iterations for solution (k^2 * GPS data size): {size}".format(size=len(self.gps_data) * self.k ** 2))
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
        T = len(self.gps_data)
        R = len(self.network)

        obs_prob = sp.lil_matrix((T, R), dtype=np.float64)

        for t in range(T):
            # get the gps point
            gps_point = self.gps_data.iloc[t]
            point_cord = (gps_point['longitude'], gps_point['latitude'])

            # find the nearest k roads
            nearest = list(self.rtree.nearest(point_cord, self.k))
            # print(len(nearest))
            # store the closest roads to gps point t
            self.close_roads[t] = nearest

            # loop through nearest roads
            for segment_id in nearest:
                # calculate distance
                closest_point = get_closest_point(gps_point, self.network.iloc[segment_id])
                distance = get_distance(gps_point, closest_point)

                # fill matrices
                self.cp_mat[t, segment_id, 0] = closest_point[0]
                self.cp_mat[t, segment_id, 1] = closest_point[1]
                self.cd_mat[t, segment_id] = distance

                if distance <= self.dist_tolerance:
                    # make a list of all seen matrices
                    if segment_id not in self.seen:
                        self.seen.add(segment_id)
                    obs_prob[t, segment_id] = norm.pdf(distance, loc=0, scale=self.scale)

        # Normalize each row of the obs_prob array
        # obs_prob /= obs_prob.sum(axis=1)

        return obs_prob.tocsc()

    def calc_trans_prob(self):
        """
        Calculates the transition probabilities using the HMM algorithm.

        Returns:
        - trans_prob: a 3D numpy array of transition probabilities
        """
        T = len(self.gps_data)
        R = len(self.network)

        # trans_mat = np.zeros((T - 1, R, R), dtype=np.float16)
        trans_mat = {}

        # probability we transition from r_i to r_j
        for t in range(T - 1):
            sparse_matrix = sp.lil_matrix((R, R), dtype=np.float64)
            # get gps points
            z_t = self.gps_data.iloc[t]
            z_t1 = self.gps_data.iloc[t + 1]

            for r_i in self.close_roads[t]:
                # if the probability of being on road r_i at time t is zero, we don't have to calculate the trans prob
                if self.obs_prob[t, r_i] != 0:
                    for r_j in self.close_roads[t]:
                        # if the probability of being on road r_j  at time t+1 is zero, we don't have to calculate the
                        # trans prob
                        if self.obs_prob[t + 1, r_j] != 0:
                            if r_i == r_j:
                                d_t = get_distance(z_t, z_t1)
                            else:
                                # should be graph route distance, decent proxy for now
                                route_dist = abs(self.cp_mat[t, r_i, 0] - self.cp_mat[t, r_j, 0]) + abs(
                                    self.cp_mat[t, r_i, 1] - self.cp_mat[t, r_j, 1])
                                d_t = get_distance(z_t, z_t1) - route_dist

                            sparse_matrix[r_i, r_j] = (1 / self.beta) * np.exp(-d_t / self.beta)

            trans_mat[t] = sparse_matrix.tocsr()

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
          being observed from each state
        """

        # Determine the number of observations and states
        n_obs = self.obs_prob.shape[0]
        n_states = self.obs_prob.shape[1]

        # Initialize the viterbi and backpointer tables
        viterbi_table = np.zeros((n_obs, n_states), dtype=self.obs_prob.dtype)
        backpointer_table = np.zeros((n_obs, n_states), dtype=np.float64)

        # Initialize the first row of the viterbi table with the start probabilities
        # print(self.obs_prob[0, :])
        # print( self.obs_prob[0, 1])
        for idx in range(0, n_states):
            if self.obs_prob[0, idx] != 0:
                viterbi_table[0, idx] = self.obs_prob[0, idx]

        # Loop through the remaining observations
        for t in range(1, n_obs):
            # Loop through the states
            for r in self.seen:

                # Calculate the scores for transitioning to this state from each previous state
                trans_scores = viterbi_table[t - 1, :] * self.trans_prob[t - 1][:, r]

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

    def filter_tree(self, rtree):
        df = self.gps_data
        min_lat, min_lat_long = df.loc[df['latitude'].idxmin()][['latitude', 'longitude']]
        max_lat, max_lat_long = df.loc[df['latitude'].idxmax()][['latitude', 'longitude']]

        min_long_lat, min_long = df.loc[df['longitude'].idxmin()][['latitude', 'longitude']]
        max_long_lat, max_long = df.loc[df['longitude'].idxmax()][['latitude', 'longitude']]

        buffer_m = 500
        buffer_degrees = (buffer_m / 1000) / 111.32
        buffer_degrees = 0
        minx = min_long - buffer_degrees
        maxx = max_long + buffer_degrees
        miny = min_lat - buffer_degrees
        maxy = max_lat + buffer_degrees
        bbox = (minx, maxx, miny, maxy)

        hits = list(rtree.intersection((bbox)))
        print(len(hits))
        # results = [(item.object, item.bbox) for item in hits]
        # print(len(results))
        #
        # return results
