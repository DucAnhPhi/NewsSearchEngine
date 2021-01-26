import pytest
from ...embedding.model import EmbeddingModel
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction

class TestFENetzpolitik():
    @classmethod
    def setup_class(self):
        self.embedder_DE = EmbeddingModel()
        self.embedder_EN = EmbeddingModel(lang="en")
        self.parser = ParserNetzpolitik()
        self.fe_DE = FeatureExtraction(self.embedder_DE, self.parser)
        self.fe_EN = FeatureExtraction(self.embedder_EN, self.parser)

    def test_semantic_specifity_DE(self):
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
        ss_sim = self.fe_DE.get_semantic_specifity(article_sim)
        ss_diff = self.fe_DE.get_semantic_specifity(article_diff)
        assert ss_sim < ss_diff

    def test_semantic_specifity_empty_DE(self):
        empty = {
            "keywords": []
        }
        ss = self.fe_DE.get_semantic_specifity(empty)
        assert ss == 2

    def test_semantic_specifity_one_DE(self):
        empty = {
            "keywords": ["test"]
        }
        ss = self.fe_DE.get_semantic_specifity(empty)
        assert ss == 2

    def test_semantic_specifity_EN(self):
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
        ss_sim = self.fe_EN.get_semantic_specifity(article_sim)
        ss_diff = self.fe_EN.get_semantic_specifity(article_diff)
        assert ss_sim < ss_diff

    def test_semantic_specifity_empty_EN(self):
        empty = {
            "keywords": []
        }
        ss = self.fe_EN.get_semantic_specifity(empty)
        assert ss == 2

    def test_semantic_specifity_one_EN(self):
        empty = {
            "keywords": ["test"]
        }
        ss = self.fe_EN.get_semantic_specifity(empty)
        assert ss == 2