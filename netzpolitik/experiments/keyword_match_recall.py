import os
import json
import argparse
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
        self.except_count = 0

        with open(judgement_list_path, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                try:
                    query_article_es = self.es.get(
                        index = self.index,
                        id = judgement["id"]
                    )
                    query = get_query_func(query_article_es)
                    if not query:
                        continue
                    self.count += 1
                    results = (self.es.search(
                        size = size,
                        index = self.index,
                        body = {
                            "query": {
                                "multi_match": {
                                    "fields": [ "title", "body" ],
                                    "query": query,
                                    "analyzer": "german",
                                    "operator": "or"
                                }
                            }
                        }
                    ))["hits"]["hits"]
                    result_ids = [res["_id"] for res in results]
                    recall = 0.
                    self.retrieval_count_avg += len(result_ids)
                    for res_id in result_ids:
                        if res_id == judgement["id"]:
                            continue
                        if res_id in judgement["references"]:
                            recall += 1
                    recall /= len(judgement["references"])
                    self.recall_avg += recall
                    if recall < self.min_recall:
                        self.min_recall = recall
                    if recall > self.max_recall:
                        self.max_recall = recall
                except Exception as e:
                    self.except_count += 1
                    print(e)
                    # query article not found
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")
        print(f"Min Recall: {self.min_recall}")
        print(f"Max Recall: {self.max_recall}")
        print(f"exceptions: {self.except_count}")

if __name__ == "__main__":
    index = "netzpolitik"
    p = argparse.ArgumentParser(description='Run netzpolitik.org keyword match recall experiments')
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

    parser = ParserNetzpolitik(es)
    judgement_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_netzpolitik.jsonl"

    print("Netzpolitik Keyword Match Retrieval Experiment")

    def get_query_from_annotated_keywords(es_doc):
        keywords = es_doc["_source"]["keywords"]
        return " ".join(keywords)
    exp = KeywordsMatchExperiment(es, args.index_name, 200, get_query_from_annotated_keywords, judgement_location)
    print("----------------------------------------------------------------")
    print("Query by string query with concatenated pre-annotated keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")

    def get_query_from_tf_idf_keywords(es_doc):
        keywords = parser.get_keywords_tf_idf(args.index_name, es_doc["_id"])
        return " ".join(keywords)
    exp = KeywordsMatchExperiment(es, args.index_name, 200, get_query_from_tf_idf_keywords, judgement_location)
    print("----------------------------------------------------------------")
    print("Query by string query with concatenated extracted tf-idf keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")

    def get_query_from_annotated_and_tf_idf_keywords(es_doc):
        annotated = es_doc["_source"]["keywords"]
        extracted = parser.get_keywords_tf_idf(args.index_name, es_doc["_id"])
        return " ".join(annotated + extracted)
    exp = KeywordsMatchExperiment(es, args.index_name, 200, get_query_from_annotated_and_tf_idf_keywords, judgement_location)
    print("----------------------------------------------------------------")
    print("Query by string query with concatenated annotated and extracted tf-idf keywords.")
    exp.print_stats()
    print("----------------------------------------------------------------")