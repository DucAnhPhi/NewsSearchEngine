#!/usr/bin/env python

from elasticsearch import helpers
from elasticsearch import Elasticsearch, TransportError
import argparse
import gzip
import json
import re
import sys
import traceback
from tqdm import tqdm
from feature_extraction import FeatureExtraction
from embedding.model import EmbeddingModel

if __name__ == "__main__":
    embedder = EmbeddingModel()
    fe = FeatureExtraction(embedder)
    parser = argparse.ArgumentParser(description='Index netzpolitik.org articles to ElasticSearch')
    parser.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    parser.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    parser.add_argument('--index_name', default='netzpolitik', help='index name')
    parser.add_argument('--create', action='store_true')

    args = parser.parse_args()
        
    es = Elasticsearch(hosts=[{"host": args.host, "port": args.port}],
                    retry_on_timeout=True, max_retries=10)
    settings = {
        'settings': {
            'index': {
                # Optimize for loading; this gets reset when we're done.
                'refresh_interval': '-1',
                'number_of_shards': '5',
                'number_of_replicas': '0'
            },
            # Set up a custom unstemmed analyzer.
            'analysis': {
                "filter": {
                    "german_stemmer": {
                        "type": "stemmer",
                        "language": "light_german"
                    }
                },
                "analyzer": {
                    "german_analyzer": {
                    "tokenizer":  "standard",
                    "filter": [
                        "lowercase",
                        "german_normalization",
                        "german_stemmer"
                    ]
                    }
                }
            }
        },
        'mappings': {
            'properties': {
                'title': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'german_analyzer'
                },
                'subtitle': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'german_analyzer'
                },
                'published': {
                    'type': 'date',
                    'format': 'dd-MM-yyyy'
                },
                'body': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'german_analyzer'
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

                # add feature: semantic_specifity, which is the mean of pairwise cosine distances of text embeddings
                #article['semantic_specifity'] = fe.get_semantic_specifity(article)

                data_dict['_source'] = article

            except Exception:
                traceback.print_exc(file=sys.stdout)
                quit()
            yield data_dict

    print("Counting...")
    with open('data/netzpolitik.jsonl', 'r') as f:
        lines = 0
        for line in f:
            lines += 1

    print("Indexing...")
    with open('data/netzpolitik.jsonl', 'r') as f:
        helpers.bulk(es, doc_generator(f, lines), request_timeout=30)

    es.indices.put_settings(index=args.index_name,
                            body={'index': { 'refresh_interval': '1s',
                                            'number_of_replicas': '1',
                            }})
