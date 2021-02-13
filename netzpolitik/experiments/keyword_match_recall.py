import os
import json
from elasticsearch import Elasticsearch
from ..parser import ParserNetzpolitik

class KeywordsMatchExperiment():
    def __init__(self, size, keywords_tf_idf = False):
        self.es = Elasticsearch()
        self.parser = ParserNetzpolitik(self.es)
        self.index = "netzpolitik"
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.

        judgement_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_netzpolitik.jsonl"
        with open(judgement_location, "r", encoding="utf-8") as f:
            for line in f:
                judgment = json.loads(line)
                try:
                    query_article = (self.es.get(
                        index=self.index,
                        id=judgment["id"]
                    ))["_source"]
                    results = []
                    query = " OR ".join(query_article["keywords"])
                    if keywords_tf_idf:
                        query = " OR ".join(self.parser.get_keywords_tf_idf(self.index, judgment["id"]))
                    if len(query) == 0:
                            continue
                    self.count += 1
                    results = (self.es.search(
                        size = size,
                        index = self.index,
                        body = {
                            "query": {
                                "query_string": {
                                    "fields": [ "title", "body" ],
                                    "query": query,
                                    "analyzer": "german"
                                }
                            }
                        }
                    ))["hits"]["hits"]
                    recall = 0.
                    self.retrieval_count_avg += len(results)
                    for res in results:
                        if res["_id"] != judgment["id"] and res["_id"] in query_article["references"]:
                            recall += 1
                    recall /= len(query_article["references"])
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
    exp = KeywordsMatchExperiment(200, True)
    print("----------------------------------------------------------------")
    print("Index articles in Elasticsearch.")
    print("Query by string query with concatenated pre-annotated keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")

    #exp = KeywordsMatchExperiment(200, True)
    #print("----------------------------------------------------------------")
    #print("Index articles in Elasticsearch.")
    #print("Query by string query with concatenated tf-idf keywords.")
    #exp.print_stats()
    #print("----------------------------------------------------------------")