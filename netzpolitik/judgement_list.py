import argparse
import json
import os
import re
from collections import defaultdict
from elasticsearch import Elasticsearch

pattern = r"^https://netzpolitik\.org/20[0-9][0-9]/.+[^#respond]$"


if __name__ == "__main__":
    index = "netzpolitik"
    p = argparse.ArgumentParser(description='Generate netzpolitik.org judgement list')
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

    keys = set()
    ref_count = 0
    id_count = 0

    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"

    with open(f'{data_location}/netzpolitik.jsonl', 'r', encoding="utf-8") as fi:
        for line in fi:
            js = json.loads(line)
            keys.add(js["url"])

    with open(f'{data_location}/netzpolitik.jsonl', 'r', encoding="utf-8") as read_fi:
        with open(f'{data_location}/judgement_list_netzpolitik.jsonl', 'w', encoding="utf-8") as out_fi:
            for line in read_fi:
                js = json.loads(line)
                try:
                    judgement_id = (es.search(
                        size = 1,
                        index = index,
                        body = {
                            "query": {
                                "term": {
                                    "url": {
                                        "value": js["url"]
                                    }
                                }
                            }
                        }
                    ))["hits"]["hits"][0]["_id"]
                    judgement = {
                        'id': judgement_id,
                        'references': []
                    }

                    for ref in js['references']:
                        # make sure reference is present in dataset
                        if re.match(pattern, ref) and ref in keys:
                            ref_count += 1
                            ref_id = (es.search(
                                size = 1,
                                index = index,
                                body = {
                                    "query": {
                                        "term": {
                                            "url": {
                                                "value": ref
                                            }
                                        }
                                    }
                                }
                            ))["hits"]["hits"][0]["_id"]
                            judgement['references'].append(ref_id)

                    if len(judgement['references']) > 0:
                        id_count += 1
                        json.dump(judgement, out_fi)
                        out_fi.write('\n')

                except Exception:
                    print("error")
    
    print("docs count:", id_count)
    print("ref count:", ref_count)

# docs count: 7691
# ref count: 26290