import os
import json
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from ..parser import ParserWAPO
from pprint import pprint

class KeywordsMatchExperiment():
    def __init__(self, es, parser, index, size, judgement_list_path):
        self.es = es
        self.parser = parser
        self.index = index
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.
        self.min_recall = 1.
        self.max_recall = 0.
        self.rel_cutoff = 4

        with open(judgement_list_path, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                # apply relevance cutoff
                relevant_articles = [ref for ref in judgement["references"] if int(ref["exp_rel"]) >= self.rel_cutoff]
                if len(relevant_articles) == 0:
                    continue
                try:
                    query_keywords = " OR ".join(self.parser.get_keywords_tf_idf(self.index, judgement["id"]))
                    self.count += 1
                    results = (self.es.search(
                        size = size,
                        index = self.index,
                        body = {
                            "query": {
                                "query_string": {
                                    "fields": [ "title", "text" ],
                                    "query": query_keywords
                                }
                            }
                        }
                    ))["hits"]["hits"]
                    recall = 0.
                    self.retrieval_count_avg += len(results)
                    for res in results:
                        if res["_id"] == judgement["id"]:
                            continue
                        for ref in relevant_articles:
                            if ref["id"] == res["_id"]:
                                recall += 1
                                break
                    recall /= len(relevant_articles)
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
    parser = ParserWAPO(es)
    index = "wapo_clean"
    judgement_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_wapo.jsonl"

    exp = KeywordsMatchExperiment(es, parser, index, 200, judgement_location)
    print("WAPO Keyword Match Retrieval Experiment")
    print("----------------------------------------------------------------")
    print("Query by string query with concatenated extracted tf-idf keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")

    # MLT results:
    # Keyword match query recall avg: 0.632946
    # Retrieval Count Avg: 100

    # Custom query, using termvectors:
    # Keyword match query recall avg: 0.636238
    # Retrieval Count Avg: 100

    # Custom query, using termvectors:
    # Keyword match query recall avg: 0.678989
    # Retrieval Count Avg: 150

    # Custom query, using termvectors:
    # Keyword match query recall avg: 0.719923
    # Retrieval Count Avg: 200

    # Custom query, using termvectors:
    # Keyword match query recall avg: 0.742124
    # Retrieval Count Avg: 250

    # Custom query, using termvectors:
    # Keyword match query recall avg: 0.752354
    # Retrieval Count Avg: 300

    # Custom query, using termvectors:
    # Keyword match query recall avg: 0.758316
    # Retrieval Count Avg: 350

    # Custom query, using termvectors:
    # Keyword match query recall avg: 0.764501
    # Retrieval Count Avg: 400