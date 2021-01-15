import os
import json
from elasticsearch import Elasticsearch

class KeywordsMatchExperiment():
    def __init__(self):
        self.es = Elasticsearch()
        self.index = "netzpolitik"
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.

        judgement_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_netzpolitik.jsonl"
        with open(judgement_location, "r") as f:
            for line in f:
                judgment = json.loads(line)
                try:
                    query_article = (self.es.get(
                        index=self.index,
                        id=judgment["id"]
                    ))["_source"]
                    query = " ".join(query_article["keywords"])
                    if len(query) == 0:
                        continue
                    self.count += 1
                    results = self.es.search(
                        index = self.index,
                        body = {
                            "query": {
                                "multi_match": {
                                    "fields": [ "title", "subtitle", "body" ],
                                    "query": query
                                }
                            }
                        }
                    )
                    recall = 0.
                    self.retrieval_count_avg += len(results["hits"]["hits"])
                    for res in results["hits"]["hits"]:
                        if res["_id"] in query_article["references"]:
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
    # Keyword match query recall avg: 0.089381
    # Retrieval Count Avg: 10.0515
    exp = KeywordsMatchExperiment()
    print("----------------------------------------------------------------")
    print("Index articles in Elasticsearch.")
    print("Query by multi match query with concatenated keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")