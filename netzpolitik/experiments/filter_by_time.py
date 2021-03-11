import argparse
import json
import os
from datetime import *
from elasticsearch import Elasticsearch

if __name__ == "__main__":
    index = "netzpolitik"

    p = argparse.ArgumentParser(description='Run netzpolitik.org filter by time experiments')
    p.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    p.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    p.add_argument('--index_name', default=index, help='index name')
    p.add_argument('--user', default=None, help='ElasticSearch user')
    p.add_argument('--secret', default=None, help="ElasticSearch secret")
    p.add_argument('--device', default="cpu", help="(CUDA) device for pytorch")

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

    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    judgement_list_path = f"{data_location}/judgement_list_netzpolitik.jsonl"

    ref_count = 0
    future_count = 0
    not_found_ref_articles = set()
    not_found_query_articles = set()

    with open(judgement_list_path, "r", encoding="utf-8") as f:
        for line in f:
            judgement = json.loads(line)
            ref_count += len(judgement["references"])
            try:
                query_article_es = es.get(
                    index = index,
                    id = judgement["id"]
                )
                for ref in judgement["references"]:
                    try:
                        ref_article_es = es.get(
                            index = index,
                            id = ref
                        )
                        dRef, mRef, yRef = [int(el) for el in ref_article_es["_source"]["published"].split("-")]
                        dateRef = date(yRef, mRef, dRef)
                        dQ, mQ, yQ = [int(el) for el in query_article_es["_source"]["published"].split("-")]
                        dateQ = date(yQ, mQ, dQ)
                        if dateRef > dateQ:
                            future_count += 1
                    except Exception as err:
                        print(f"Not found ref article: {err}")
                        not_found_ref_articles.add(ref["id"])
                        pass
            except Exception as e:
                print(f"Not found query article: {e}")
                not_found_query_articles.add(judgement["id"])
                continue
    
    print(f"Number of relevant articles published after reference article: {future_count}")
    print(f"Number of all relevant articles: {ref_count}")
    print(f"Ratio: {future_count/ref_count}")
    print(f"Number of not found relevant articles: {len(not_found_ref_articles)}")
    print(f"Number of not found query articles: {len(not_found_query_articles)}")