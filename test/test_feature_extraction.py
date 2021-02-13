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
        article_sim = {
            "keywords": [
                "Huhn",
                "Ei",
                "Vogel",
                "Geflügel"
            ]
        }
        article_diff = {
            "keywords": [
                "Code",
                "Geflügel",
                "Siebträger",
                "Donald Trump"
            ]
        }
        ss_sim = self.fe_DE.get_keywords_similarity(article_sim)
        ss_diff = self.fe_DE.get_keywords_similarity(article_diff)
        assert ss_sim < ss_diff

    def test_keywords_similarity_empty_DE(self):
        empty = {
            "keywords": []
        }
        ss = self.fe_DE.get_keywords_similarity(empty)
        assert ss == 0

    def test_keywords_similarity_one_DE(self):
        empty = {
            "keywords": ["test"]
        }
        ss = self.fe_DE.get_keywords_similarity(empty)
        assert ss == 0

    def test_keywords_similarity_EN(self):
        article_sim = {
            "keywords": [
                "Chicken",
                "Egg",
                "Bird",
                "Poultry"
            ]
        }
        article_diff = {
            "keywords": [
                "Code",
                "Poultry",
                "Portafilter",
                "Donald Trump"
            ]
        }
        ss_sim = self.fe_EN.get_keywords_similarity(article_sim)
        ss_diff = self.fe_EN.get_keywords_similarity(article_diff)
        assert ss_sim < ss_diff

    def test_keywords_similarity_empty_EN(self):
        empty = {
            "keywords": []
        }
        ss = self.fe_EN.get_keywords_similarity(empty)
        assert ss == 0

    def test_keywords_similarity_one_EN(self):
        empty = {
            "keywords": ["test"]
        }
        ss = self.fe_EN.get_keywords_similarity(empty)
        assert ss == 0