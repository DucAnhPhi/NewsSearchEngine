import pytest
from elasticsearch import Elasticsearch

class TestRetrieval():
    @classmethod
    def setup_class(self):
        self.es = Elasticsearch()
        self.index = "netzpolitik"
    
    def test_get_article_by_id_es(self):
        article = self.es.get(index=self.index, id="https://netzpolitik.org/2020/antiviren-amazon-und-ai/")
        print(article["_source"]["keywords"])
        assert len(article["_source"]["keywords"]) != 0

    def test_compute_recall(self):
        recall = 0
        results = ["a", "b", "c", "d", "e", "f", "g"]
        relevant = ["a", "g", "z"]
        for res in results:
            if res in relevant:
                recall += 1
        recall /= len(relevant)
        expected_recall = 2/3
        assert expected_recall == recall
    
    def test_query_articles_es(self):
        keywords = [
            "EU-Urheberrechtsreform",
            "EuGH",
            "Europäische Kommission",
            "Europäischer Gerichtshof",
            "Leistungsschutzrecht für Presseverleger",
            "LSR",
            "Notifizierung",
            "Österreich",
            "Urheberrecht",
            "Zeitplan",
            "Bündnis 90/Die Grünen"
        ]
        results = self.es.search(
            index = self.index,
            body = {
                "query": {
                    "multi_match": {
                        "fields": [ "title", "subtitle", "body" ],
                        "query": " ".join(keywords)
                    }
                }
            }
        )
        result_ids = [res["_id"] for res in results["hits"]["hits"]]
        assert len(result_ids) != 0