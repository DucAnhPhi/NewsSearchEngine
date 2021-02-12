import os
import json
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl.query import MoreLikeThis
from ..parser import ParserWAPO
from pprint import pprint

class KeywordsMatchExperiment():
    def __init__(self):
        self.es = Elasticsearch()
        self.parser = ParserWAPO(self.es)
        self.index = "wapo_clean"
        self.s = Search(using=self.es, index=self.index)
        # specify search size to 100
        self.s = self.s[:100]
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.
        self.rel_cutoff = 4

        judgement_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_wapo.jsonl"
        with open(judgement_location, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                # apply relevance cutoff
                filtered = [ref for ref in judgement["references"] if int(ref["exp_rel"]) >= self.rel_cutoff]
                if len(filtered) == 0:
                    continue
                try:
                    self.count += 1
                    #results = self.s.query(
                    #    MoreLikeThis(
                    #        like={'_id': judgement["id"], '_index': self.index},
                    #        fields=["title", "text"]
                    #    )
                    #).execute()
                    query_keywords = " OR ".join(self.parser.get_keywords_tf_idf(self.index, judgement["id"]))
                    results = (self.es.search(
                        size = 200,
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
                        for ref in filtered:
                            if ref["id"] == res["_id"]:
                                recall += 1
                                break
                    recall /= len(filtered)
                    print(recall, judgement["id"])
                    self.recall_avg += recall
                except:
                    print("err")
                    self.count -= 1
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")

if __name__ == "__main__":
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
    exp = KeywordsMatchExperiment()
    print("----------------------------------------------------------------")
    print("Outlet: WAPO")
    print("Index articles in Elasticsearch.")
    print("Perform 'More Like This' Query")
    exp.print_stats()
    print("----------------------------------------------------------------")