import pytest
import json
from ..embedding.model import EmbeddingModel
from ..feature_extraction import FeatureExtraction
import numpy as np

class TestFeatureExtraction():
    @classmethod
    def setup_class(self):
        self.embedder_DE = EmbeddingModel()
        self.embedder_EN = EmbeddingModel(lang="en")
        self.fe_DE = FeatureExtraction(self.embedder_DE, None)
        self.fe_EN = FeatureExtraction(self.embedder_EN, None)

    def test_mean_of_pairwise_cosine_distances(self):
        ems = np.array([
            [-1,1,1],
            [-11,3,9],
            [22,0,8]
        ], dtype=float)
        assert abs(0.9770 - FeatureExtraction.mean_of_pairwise_cosine_distances(ems)) < 1e-4

    def test_keywords_similarity_DE(self):
        keywords_sim = [
            "Huhn",
            "Ei",
            "Vogel",
            "Geflügel"
        ]
        keywords_diff = [
            "Code",
            "Geflügel",
            "Siebträger",
            "Donald Trump"
        ]
        ss_sim = self.fe_DE.get_keywords_similarity(keywords_sim)
        ss_diff = self.fe_DE.get_keywords_similarity(keywords_diff)
        assert ss_sim < ss_diff

    def test_keywords_similarity_empty_DE(self):
        empty = []
        ss = self.fe_DE.get_keywords_similarity(empty)
        assert ss == 0

    def test_keywords_similarity_one_DE(self):
        empty = ["test"]
        ss = self.fe_DE.get_keywords_similarity(empty)
        assert ss == 0

    def test_keywords_similarity_EN(self):
        keywords_sim = [
            "Chicken",
            "Egg",
            "Bird",
            "Poultry"
        ]
        keywords_diff = [
            "Code",
            "Poultry",
            "Portafilter",
            "Donald Trump"
        ]
        ss_sim = self.fe_EN.get_keywords_similarity(keywords_sim)
        ss_diff = self.fe_EN.get_keywords_similarity(keywords_diff)
        assert ss_sim < ss_diff

    def test_keywords_similarity_empty_EN(self):
        empty = []
        ss = self.fe_EN.get_keywords_similarity(empty)
        assert ss == 0

    def test_keywords_similarity_one_EN(self):
        empty = ["test"]
        ss = self.fe_EN.get_keywords_similarity(empty)
        assert ss == 0