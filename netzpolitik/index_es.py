#!/usr/bin/env python
from elasticsearch import helpers
from elasticsearch import Elasticsearch, TransportError
import argparse
import gzip
import json
import re
import sys
import os
import traceback
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Index netzpolitik.org articles to ElasticSearch')
    p.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    p.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    p.add_argument('--index_name', default='netzpolitik', help='index name')
    p.add_argument('--create', action='store_true')

    args = p.parse_args()
        
    es = Elasticsearch(hosts=[{"host": args.host, "port": args.port}],
                    retry_on_timeout=True, max_retries=10)
    settings = {
        'settings': {
            'index': {
                # Optimize for loading; this gets reset when we're done.
                'refresh_interval': '-1',
                'number_of_shards': '5',
                'number_of_replicas': '0'
            }
        },
        'mappings': {
            'properties': {
                'title': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'german',
                    'term_vector': 'yes'
                },
                'subtitle': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'german',
                    'term_vector': 'yes'
                },
                'published': {
                    'type': 'date',
                    'format': 'dd-MM-yyyy'
                },
                'body': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'german',
                    'term_vector': 'yes'
                },
                'raw_body': {
                    'type': 'object',
                    'enabled': False
                }
            }
        }
    }

    if args.create or not es.indices.exists(index=args.index_name):
        try:
            es.indices.create(index=args.index_name, body=settings)
        except TransportError as e:
            print(e.info)
            sys.exit(-1)

    def doc_generator(f, num_docs):
        for line in tqdm(f, total=num_docs):
            js = json.loads(line)
            try:
                data_dict = {
                    "_index": args.index_name,
                    "_type": '_doc',
                    "_id": js['id'],
                }

                article = js.copy()
                del article['id']

                data_dict['_source'] = article

            except Exception:
                traceback.print_exc(file=sys.stdout)
                quit()
            yield data_dict

    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    print("Counting...")
    with open(f"{data_location}/netzpolitik.jsonl", "r", encoding="utf-8") as f:
        lines = 0
        for line in f:
            lines += 1

    print("Indexing...")
    with open(f"{data_location}/netzpolitik.jsonl", "r", encoding="utf-8") as f:
        helpers.bulk(es, doc_generator(f, lines), request_timeout=30)

    es.indices.put_settings(index=args.index_name,
                            body={'index': { 'refresh_interval': '1s',
                                            'number_of_replicas': '1',
                            }})
