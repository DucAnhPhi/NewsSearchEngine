import os
import json
from elasticsearch import Elasticsearch
from ..parser import ParserNetzpolitik

class KeywordsMatchCombinedExperiment():
    def __init__(self, size):
        self.es = Elasticsearch()
        self.parser = ParserNetzpolitik(self.es)
        self.index = "netzpolitik"
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.

        judgement_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_netzpolitik.jsonl"
        with open(judgement_location, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                try:
                    query_article = self.es.get(
                        index = self.index,
                        id = judgement["id"]
                    )
                    results = []
                    tf_idf_query = " OR ".join(self.parser.get_keywords_tf_idf(self.index, judgement["id"]))
                    self.count += 1
                    results = (self.es.search(
                        size = size,
                        index = self.index,
                        body = {
                            "query": {
                                "query_string": {
                                    "fields": [ "title", "body" ],
                                    "query": tf_idf_query,
                                    "analyzer": "german"
                                }
                            }
                        }
                    ))["hits"]["hits"]
                    annotated_query = " OR ".join(query_article["_source"]["keywords"])
                    if annotated_query:
                        annotated_results = (self.es.search(
                            size = size,
                            index = self.index,
                            body = {
                                "query": {
                                    "query_string": {
                                        "fields": [ "title", "body" ],
                                        "query": annotated_query,
                                        "analyzer": "german"
                                    }
                                }
                            }
                        ))["hits"]["hits"]
                        results = results + [res for res in annotated_results if not any(res["_id"] == r["_id"] for r in results)]
                    recall = 0.
                    self.retrieval_count_avg += len(results)
                    for res in results:
                        if res["_id"] != judgement["id"] and res["_id"] in judgement["references"]:
                            recall += 1
                    recall /= len(judgement["references"])
                    self.recall_avg += recall
                except:
                    self.count -= 1
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")

if __name__ == "__main__":
    exp = KeywordsMatchCombinedExperiment(100)
    print("----------------------------------------------------------------")
    print("Index articles in Elasticsearch.")
    print("Query by string query with concatenated pre-annotated and extracted tf-idf keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")