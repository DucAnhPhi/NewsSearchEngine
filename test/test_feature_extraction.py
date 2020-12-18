import sys
sys.path.append("..")
import unittest
import json
from feature_extraction import FeatureExtraction
import numpy as np

class TestFeatureExtraction(unittest.TestCase):

    def test_mean_of_pairwise_cosine_distances(self):
        ems = np.array([
            [-1,1,1],
            [-11,3,9],
            [22,0,8]
        ], dtype=float)
        self.assertTrue(abs(0.9770 - FeatureExtraction.mean_of_pairwise_cosine_distances(ems)) < 1e-4)