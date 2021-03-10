import os
import json
import argparse
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
        self.exception_count = 0

        with open(judgement_list_path, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                # apply relevance cutoff
                relevant_articles = [ref for ref in judgement["references"] if int(ref["exp_rel"]) >= self.rel_cutoff]
                if len(relevant_articles) == 0:
                    self.count += 1
                    self.min_recall = 0
                    continue
                try:
                    keywords = self.parser.get_keywords_tf_idf(self.index, judgement["id"])
                    if not keywords:
                        print(f"no keywords found for id: {judgement['id']}")
                        self.exception_count += 1
                        continue
                    self.count += 1
                    query_keywords = " OR ".join(keywords)
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
                except Exception as e:
                    # query article not found
                    self.exception_count += 1
                    print(e)
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")
        print(f"Min Recall: {self.min_recall}")
        print(f"Max Recall: {self.max_recall}")
        print(f"Exception Count: {self.exception_count}")

if __name__ == "__main__":
    index = "wapo_clean"

    p = argparse.ArgumentParser(description='Run Washington Post keyword match retrieval experiments')
    p.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    p.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    p.add_argument('--index_name', default=index, help='index name')
    p.add_argument('--user', default=None, help='ElasticSearch user')
    p.add_argument('--secret', default=None, help="ElasticSearch secret")

    args = p.parse_args()

    es = None

    if args.user and args.secret:
        es = Elasticsearch(
            hosts = [{"host": args.host, "port": args.port}],
            http_auth=(args.user, args.secret),
            scheme="https",
            retry_on_timeout=True,
            max_retries=10
        )
    else:
        es = Elasticsearch(
            hosts=[{"host": args.host, "port": args.port}],
            retry_on_timeout=True,
            max_retries=10
        )

    parser = ParserWAPO(es)
    judgement_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_wapo.jsonl"

    exp = KeywordsMatchExperiment(es, parser, index, 300, judgement_location)
    print("WAPO Keyword Match Retrieval Experiment")
    print("----------------------------------------------------------------")
    print("Query by string query with concatenated extracted tf-idf keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")