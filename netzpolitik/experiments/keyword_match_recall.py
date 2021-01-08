import os
import json
from elasticsearch import Elasticsearch

if __name__ == "__main__":
    # Keyword match query recall avg: 0.08865350894055515
    es = Elasticsearch()
    index = "netzpolitik"
    count = 0
    recall_avg = 0.
    with open(f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_netzpolitik.jsonl", "r") as f:
        for line in f:
            judgment = json.loads(line)
            try:
                query_article = (es.get(
                    index=index,
                    id=judgment["id"]
                ))["_source"]
                query = " ".join(query_article["keywords"])
                if len(query) == 0:
                    continue
                count += 1
                results = es.search(
                index = index,
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
                for res in results["hits"]["hits"]:
                    if res["_id"] in query_article["references"]:
                        recall += 1
                recall /= len(query_article["references"])
                recall_avg += recall
            except:
                count -= 1
                continue
    recall_avg /= count
    print(f"Keyword match query recall avg: {recall_avg}")