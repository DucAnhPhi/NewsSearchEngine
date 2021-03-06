import argparse
import json
import os
from elasticsearch import Elasticsearch
from ..parser import ParserNetzpolitik

class KeywordsMatchCombinedExperiment():
    def __init__(self, es, index, size):
        self.es = es
        self.parser = ParserNetzpolitik(self.es)
        self.index = index
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.
        self.except_count = 0

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
                    tf_idf_query = " ".join(self.parser.get_keywords_tf_idf(self.index, judgement["id"]))
                    self.count += 1
                    results = (self.es.search(
                        size = size,
                        index = self.index,
                        body = {
                            "query": {
                                "multi_match": {
                                    "fields": [ "title", "body" ],
                                    "query": tf_idf_query,
                                    "analyzer": "german",
                                    "operator": "or"
                                }
                            }
                        }
                    ))["hits"]["hits"]
                    annotated_query = " ".join(query_article["_source"]["keywords"])
                    if annotated_query:
                        annotated_results = (self.es.search(
                            size = size,
                            index = self.index,
                            body = {
                                "query": {
                                    "multi_match": {
                                        "fields": [ "title", "body" ],
                                        "query": annotated_query,
                                        "analyzer": "german",
                                        "operator": "or"
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
                except Exception as e:
                    self.except_count += 1
                    print(e)
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")
        print(f"Exception Count: {self.except_count}")

if __name__ == "__main__":
    index = "netzpolitik"
    p = argparse.ArgumentParser(description='Run netzpolitik.org keyword match recall combined results experiments')
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

    print("----------------------------------------------------------------")
    print("Index articles in Elasticsearch.")
    print("Query by multi-match query with concatenated pre-annotated and extracted tf-idf keywords.")
    exp = KeywordsMatchCombinedExperiment(es, index, 100)
    exp.print_stats()
    print("----------------------------------------------------------------")