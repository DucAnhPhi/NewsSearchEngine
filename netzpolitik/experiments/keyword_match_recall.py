import os
import json
from elasticsearch import Elasticsearch
from ..parser import ParserNetzpolitik

class KeywordsMatchExperiment():
    def __init__(self, es, index, size, get_query_func, judgement_list_path):
        self.es = es
        self.index = index
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.
        self.min_recall = 1.
        self.max_recall = 0.

        with open(judgement_list_path, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                try:
                    query_article_raw = (self.es.get(
                        index=self.index,
                        id=judgement["id"]
                    ))
                    query = get_query_func(query_article_raw)
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
                        if res["_id"] == judgement["id"]:
                            continue
                        if res["_id"] in judgement["references"]:
                            recall += 1
                    recall /= len(judgement["references"])
                    self.recall_avg += recall
                    if recall < self.min_recall:
                        self.min_recall = recall
                    if recall > self.max_recall:
                        self.max_recall = recall
                except:
                    # query article not found
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")
        print(f"Min Recall: {self.min_recall}")
        print(f"Max Recall: {self.max_recall}")

if __name__ == "__main__":
    es = Elasticsearch()
    parser = ParserNetzpolitik(es)
    index = "netzpolitik"
    judgement_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_netzpolitik.jsonl"

    print("Netzpolitik Keyword Match Retrieval Experiment")

    def get_query_from_annotated_keywords(raw):
        keywords = raw["_source"]["keywords"]
        return " OR ".join(keywords)
    exp = KeywordsMatchExperiment(es, index, 200, get_query_from_annotated_keywords, judgement_location)
    print("----------------------------------------------------------------")
    print("Query by string query with concatenated pre-annotated keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")

    def get_query_from_tf_idf_keywords(raw):
        keywords = parser.get_keywords_tf_idf(index, raw["_id"])
        return " OR ".join(keywords)
    exp = KeywordsMatchExperiment(es, index, 200, get_query_from_tf_idf_keywords, judgement_location)
    print("----------------------------------------------------------------")
    print("Query by string query with concatenated extracted tf-idf keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")

    def get_query_from_annotated_and_tf_idf_keywords(raw):
        annotated = raw["_source"]["keywords"]
        extracted = parser.get_keywords_tf_idf(index, raw["_id"])
        return " OR ".join(annotated + extracted)
    exp = KeywordsMatchExperiment(es, index, 200, get_query_from_annotated_and_tf_idf_keywords, judgement_location)
    print("----------------------------------------------------------------")
    print("Query by string query with concatenated annotated and extracted tf-idf keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")