import pytest
from ...embedding.model import EmbeddingModel
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction

class TestFENetzpolitik():
    @classmethod
    def setup_class(self):
        self.embedder = EmbeddingModel()
        self.parser = ParserNetzpolitik()
        self.fe = FeatureExtraction(self.embedder, self.parser)

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
        assert ss_sim < ss_diff

    def test_semantic_specifity_empty(self):
        empty = {
            "raw_body": '<p>This is a paragraph, however, there are no titles</p>'
        }
        ss = self.fe.get_semantic_specifity(empty)
        assert ss == 2

    def test_semantic_specifity_one(self):
        empty = {
            "raw_body": '<h2>t</h2><p>This is a paragraph, however, there are no titles</p>'
        }
        ss = self.fe.get_semantic_specifity(empty)
        assert ss == 2