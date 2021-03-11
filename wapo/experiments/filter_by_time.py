import argparse
import json
import os
from elasticsearch import Elasticsearch

if __name__ == "__main__":
    index = "wapo_clean"

    p = argparse.ArgumentParser(description='Run Washington Post filter by time experiments')
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
    judgement_list_path = f"{data_location}/judgement_list_wapo.jsonl"

    rel_cutoff = 4
    ref_count = 0
    future_count = 0
    not_found_ref_articles = set()
    not_found_query_articles = set()
    significant_relevant_count = 0
    essential_relevant_count = 0
    critical_relevant_count = 0

    with open(judgement_list_path, "r", encoding="utf-8") as f:
        for line in f:
            judgement = json.loads(line)
            # apply relevance cutoff
            relevant_articles = [ref for ref in judgement["references"] if int(ref["exp_rel"]) >= rel_cutoff]
            ref_count += len(relevant_articles)
            if len(relevant_articles) == 0:
                continue
            try:
                query_article_es = es.get(
                    index = index,
                    id = judgement["id"]
                )
                for ref in relevant_articles:
                    try:
                        ref_article_es = es.get(
                            index = index,
                            id = ref["id"]
                        )
                        if ref_article_es["_source"]["date"] > query_article_es["_source"]["date"]:
                            future_count += 1
                            if int(ref["exp_rel"]) == 4:
                                significant_relevant_count +=1
                            if int(ref["exp_rel"]) == 8:
                                essential_relevant_count += 1
                            if int(ref["exp_rel"]) == 16:
                                critical_relevant_count += 1
                    except Exception as err:
                        print(f"Not found ref article: {err}")
                        not_found_ref_articles.add(ref["id"])
                        pass
            except Exception as e:
                print(f"Not found query article: {e}")
                not_found_query_articles.add(judgement["id"])
                continue
    
    print(f"Number of relevant articles published after reference article: {future_count}")
    print(f"Out of these {future_count} articles: \n{significant_relevant_count} are significant, \n{essential_relevant_count} are essential and\n{critical_relevant_count} are critical for relevance.")
    print(f"Number of all relevant articles (after relevance cutoff): {ref_count}")
    print(f"Ratio: {future_count/ref_count}")
    print(f"Number of not found relevant articles: {len(not_found_ref_articles)}")
    print(f"Number of not found query articles: {len(not_found_query_articles)}")