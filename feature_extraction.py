from scipy.spatial.distance import cosine
import numpy as np
from itertools import combinations

class FeatureExtraction():

    @staticmethod
    def mean_of_pairwise_cosine_distances(self, embeddings) -> float:
        # all combinations of 2 rows, ignoring order and no repeated elements
        coms = combinations(embeddings, 2)
        dists = np.array([cosine(pair[0], pair[1]) for pair in coms])
        mean_dist = np.mean(dists)
        return mean_dist