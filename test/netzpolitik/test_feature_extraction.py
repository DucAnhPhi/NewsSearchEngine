import unittest
from ...embedding.model import EmbeddingModel
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction

class TestFENetzpolitik(unittest.TestCase):
    def setUp(self):
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
        self.assertTrue(ss_sim < ss_diff)