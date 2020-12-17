import sys
sys.path.append("..")
import unittest
import json
import os
from feature_extraction import FeatureExtraction
from embedding.model import EmbeddingModel
import numpy as np

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.embedder = EmbeddingModel()
        self.fe = FeatureExtraction(self.embedder)

    def test_mean_of_pairwise_cosine_distances(self):
        ems = np.array([
            [-1,1,1],
            [-11,3,9],
            [22,0,8]
        ], dtype=float)
        self.assertTrue(abs(0.9770-self.fe.mean_of_pairwise_cosine_distances(ems)) < 1e-4)

    def test_semantic_specifity(self):
        article_sim = {
            "raw_body": '''
                <h3>Huhn</h3>
                <h3>Ei</h3>
                <h3>Vogel</h3>
                <h3>Geflügel</h3>
            '''
        }
        article_diff = {
            "raw_body": '''
                <h2>Code</h2>
                <h2>Geflügel</h2>
                <h2>Siebträger</h2>
                <h2>Donald Trump</h2>
            '''
        }
        ss_sim = self.fe.get_semantic_specifity(article_sim)
        ss_diff = self.fe.get_semantic_specifity(article_diff)
        self.assertTrue(ss_sim < ss_diff)