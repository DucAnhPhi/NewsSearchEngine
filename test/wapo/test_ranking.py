import pytest
import os
import json
import numpy as np
from elasticsearch import Elasticsearch
from scipy.spatial.distance import cosine
from ...vector_storage import VectorStorage
from ...wapo.parser import ParserWAPO
from ...embedding.model import EmbeddingModel
from ...pyw_hnswlib import Hnswlib
from ...wapo.experiments.ranking import WAPORanker

class TestWapoRanking():
    @classmethod
    def setup_class(self):
        self.articles = []
        self.es = Elasticsearch()
        self.parser = ParserWAPO(self.es)
        self.index = "wapo_clean"
        file_location = f"{os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))}/data/test_articles_raw.jsonl"
        with open(file_location, "r", encoding="utf-8") as f:
            for line in f:
                self.articles.append(json.loads(line))
        self.em = EmbeddingModel(lang="en")
        self.storage = Hnswlib(space="cosine", dim=768)
        self.storage.init_index(max_elements=10, ef_construction=200, M=100)
        self.ranker = WAPORanker(self.es, self.parser, self.em, None, self.index)

    def test_same_cosine(self):
        raw = self.articles[0]
        parsed = self.parser.parse_article(raw)
        actual_k = self.parser.get_keywords_tf_idf_denormalized(self.index, raw["id"], self.parser.get_title(parsed), parsed["text"])
        emb = self.em.encode(" ".join(actual_k))
        self.storage.add_items([emb],[raw["id"]])
        query = "ahead region commuters transportation projects scheduled lanes Beltway open Travelers Road drivers new Intercounty Connector congestion routes Telegraph Bridge Metro riders trains Line maintenance stations entrance fare Avenue"
        query_emb = self.em.encode(query)
        cosine_1 = cosine(emb, query_emb)
        labels, distances = self.storage.knn_query([query_emb])
        cosine_2 = distances[0][0]
        assert int(cosine_1 * 1000) == 0
        assert int(cosine_1*1000) == int(cosine_2*1000)

    def test_feature_extraction(self):
        raw_query = self.articles[1]
        parsed_query = self.parser.parse_article(raw_query)
        query_doc = {"_id": raw_query["id"], "_source": parsed_query}
        doc_id = self.articles[0]["id"]
        actual_features = self.ranker.get_features(query_doc, doc_id)
        query_bm25_keywords = self.parser.get_keywords_tf_idf(self.index, query_doc["_id"])
        query_keywords = self.parser.get_keywords_tf_idf_denormalized(self.index, query_doc["_id"], query_doc["_source"]["title"], query_doc["_source"]["text"], keep_order=True)
        query_emb = self.em.encode(" ".join(query_keywords))
        doc_keywords = ['ahead', 'region', 'commuters', 'transportation', 'projects', 'scheduled', 'lanes', 'Beltway', 'open', 'Travelers', 'Road', 'drivers', 'new', 'Intercounty', 'Connector', 'congestion', 'routes', 'Telegraph', 'Bridge', 'Metro', 'riders', 'trains', 'Line', 'maintenance', 'stations', 'entrance', 'fare', 'Avenue']
        doc_emb = self.em.encode(" ".join(doc_keywords)) 
        expected_bm25 = self.es.explain(
            index=self.index,
            id=doc_id,
            body = {
                "query": {
                    "query_string": {
                        "fields": [ "title", "text" ],
                        "query": " OR ".join(query_bm25_keywords)
                    }
                }
            }
        )["explanation"]["value"]
        expected_cos = cosine(query_emb, doc_emb)
        expected_published_after = 1
        expected_doc_length = 7069 
        assert actual_features[0] == expected_bm25
        assert actual_features[1] == expected_cos
        assert actual_features[2] == expected_doc_length
        assert actual_features[3] == expected_published_after

    def test_get_training_and_validation_set(self):
        data = np.arange(100)
        data = [{"id": str(el)} for el in data]
        train_data, val_data = self.ranker.get_training_and_validation_set(data)
        assert (len(train_data) + len(val_data)) == 100
        assert len(train_data) == 70
        assert len(val_data) == 30
        data_ids = [d["id"] for d in data]
        ordered = np.arange(100)
        ordered = [str(el) for el in ordered]
        assert "".join(data_ids) != "".join(ordered)

    def test_get_ranking(self):
        test_pred = np.array([-1.1, 1, -0.5, 2, 3])
        test_ids = np.array(["a", "b", "c", "d", "e"])
        ranked_test_pred, ranked_ids = self.ranker.get_ranking(test_pred, test_ids)
        expected_test_pred = [3,2,1,-0.5,-1.1]
        expected_test_ids = ["e", "d", "b", "c", "a"]
        assert len(ranked_test_pred) == len(expected_test_pred)
        assert len(test_ids) == len(expected_test_ids)
        assert all([a == b for a,b in zip(ranked_test_pred, expected_test_pred)])
        assert all([a==b for a,b in zip(ranked_ids, expected_test_ids)])