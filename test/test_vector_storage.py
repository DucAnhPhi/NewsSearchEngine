import pytest
import os
from ..vector_storage import VectorStorage
from ..embedding.model import EmbeddingModel

class TestVectorStorage():
    @classmethod
    def setup_class(self):
        self.vs_de = VectorStorage("test", 10, persist=False)
        self.vs_en = VectorStorage("test", 10, persist=False)
        self.em_de = EmbeddingModel(lang="de")
        self.em_en = EmbeddingModel(lang="en")
        self.data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"

    def test_get_k_nearest_de(self):
        file_path = f"{self.data_location}/german_words.jsonl"
        def emb_func(obj):
            return self.em_de.encode(obj["content"])
        self.vs_de.add_items_from_file(file_path, emb_func)
        actual = self.vs_de.get_k_nearest(self.em_de.encode("Technik"), 5)[0]
        actual_ids = set([list(el.keys())[0] for el in actual])
        expected_ids = { "a", "b", "c", "d", "e" }
        assert actual_ids == expected_ids

    def test_get_k_nearest_en(self):
        file_path = f"{self.data_location}/english_words.jsonl"
        def emb_func(obj):
            return self.em_en.encode(obj["content"])
        self.vs_en.add_items_from_file(file_path, emb_func)
        actual = self.vs_en.get_k_nearest(self.em_en.encode("technology"), 5)[0]
        actual_ids = set([list(el.keys())[0] for el in actual])
        expected_ids = { "a", "b", "c", "d", "e" }
        assert actual_ids == expected_ids