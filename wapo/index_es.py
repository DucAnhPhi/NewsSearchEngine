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
from .parser import ParserWAPO
from ..embedding.model import EmbeddingModel
from ..feature_extraction import FeatureExtraction

if __name__ == "__main__":
    embedder = EmbeddingModel(lang="en")
    parser = ParserWAPO()
    fe = FeatureExtraction(embedder, parser)
    p = argparse.ArgumentParser(description='Index WashingtonPost docs to ElasticSearch')
    p.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    p.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    p.add_argument('--index_name', default='wapo', help='index name')
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
            },
            # Set up a custom unstemmed analyzer.
            'analysis': {
                'analyzer': {
                    'english_stemmed': {
                        'tokenizer': 'standard',
                        'filter': [
                            'lowercase',
                            'stemmer'
                        ]
                    }
                }
            }
        },
        'mappings': {
            'properties': {
                'text': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'english_stemmed'
                },
                'title': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'english_stemmed'
                },
                'date': {
                    'type': 'date'
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
            article = parser.parse_article(js)
            if article != None:
                article["_source"]["keywords_similarity"] = fe.get_keywords_similarity(article)
                yield article

    print("Counting...")
    with open('TREC_Washington_Post_collection.v3.jl', 'r') as f:
        lines = 0
        for line in f:
            lines += 1

    print("Indexing...")
    with open('TREC_Washington_Post_collection.v3.jl', 'r') as f:
        helpers.bulk(es, doc_generator(f, lines), request_timeout=30)

    es.indices.put_settings(index=args.index_name,
                            body={'index': { 'refresh_interval': '1s',
                                            'number_of_replicas': '1',
                            }})
