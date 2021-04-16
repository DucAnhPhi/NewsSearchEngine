#!/usr/bin/env python

from elasticsearch import Elasticsearch, TransportError, helpers
import argparse
import gzip
import json
import re
import sys
import os
import json
from tqdm import tqdm
from .parser import ParserWAPO

if __name__ == "__main__":
    parser = ParserWAPO()

    # WAPO Corpus v2
    index_name_v2 = "wapo_v2"
    index_file_v2 = "TREC_Washington_Post_collection.v2.jl"

    # WAPO Corpus v3
    index_name_combined = "wapo_clean"
    index_file_v3 = "TREC_Washington_Post_collection.v3.jl"

    p = argparse.ArgumentParser(description='Index Washington Post articles to ElasticSearch')
    p.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    p.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
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

    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    stopwords_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data/english_stopwords_nltk.txt"
    stopwords = []
    with open(stopwords_location, "r", encoding="utf-8") as f:
        for line in f:
            stopwords.append(line.strip())

    settings = {
        'settings': {
            'index': {
                # Optimize for loading; this gets reset when we're done.
                'refresh_interval': '-1',
                'number_of_shards': '5',
                'number_of_replicas': '0'
            },
            "analysis": {
                "filter": {
                    "english_stop": {
                        "type":       "stop",
                        "stopwords":  stopwords
                    },
                    "english_stemmer": {
                        "type":       "stemmer",
                        "language":   "english"
                    },
                    "english_possessive_stemmer": {
                        "type":       "stemmer",
                        "language":   "possessive_english"
                    }
                },
                "analyzer": {
                    "english_custom": {
                        "tokenizer":  "standard",
                        "filter": [
                            "english_possessive_stemmer",
                            "lowercase",
                            "english_stop",
                            "english_stemmer"
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
                    'analyzer': 'english_custom',
                    'term_vector': 'yes'
                },
                'title': {
                    'type': 'text',
                    'similarity': 'BM25',
                    'analyzer': 'english_custom',
                    'term_vector': 'yes'
                },
                'date': {
                    'type': 'date'
                },
                'url': {
                    'type': 'keyword'
                }
            }
        }
    }

    if not es.indices.exists(index=index_name_v2):
        try:
            es.indices.create(index=index_name_v2, body=settings)
        except TransportError as e:
            print(e.info)
            sys.exit(-1)

    if not es.indices.exists(index=index_name_combined):
        try:
            es.indices.create(index=index_name_combined, body=settings)
        except TransportError as e:
            print(e.info)
            sys.exit(-1)

    def doc_generator_v2(f, num_docs):
        for line in tqdm(f, total=num_docs):
            raw = json.loads(line)
            article_source = parser.parse_article(raw, ignore=False)
            if article_source is not None:
                data_dict = {
                    "_index": index_name_v2,
                    "_id": raw['id'],
                }
                data_dict["_source"] = article_source
                yield data_dict

    def doc_generator_v3(f, num_docs):
        for line in tqdm(f, total=num_docs):
            raw = json.loads(line)
            article_source = parser.parse_article(raw)
            if article_source is not None:
                data_dict = {
                    "_index": index_name_combined,
                    "_id": raw['id'],
                }
                data_dict["_source"] = article_source
                yield data_dict

    print("Counting WAPO Corpus v2...")
    articles_location_v2 = f"{data_location}/{index_file_v2}"
    with open(articles_location_v2, 'r', encoding="utf-8") as f:
        lines = 0
        for line in f:
            lines += 1

    print("Indexing WAPO Corpus v2...")
    with open(articles_location_v2, 'r', encoding="utf-8") as f:
        helpers.bulk(es, doc_generator_v2(f, lines), request_timeout=30)

    es.indices.put_settings(index=index_name_v2,
                            body={'index': { 'refresh_interval': '1s',
                                            'number_of_replicas': '1',
                            }})

    print("Counting WAPO Corpus v3...")
    articles_location_v3 = f"{data_location}/{index_file_v3}"
    with open(articles_location_v3, 'r', encoding="utf-8") as f:
        lines = 0
        for line in f:
            lines += 1

    print("Indexing WAPO Corpus v3...")
    with open(articles_location_v3, 'r', encoding="utf-8") as f:
        helpers.bulk(es, doc_generator_v3(f, lines), request_timeout=30)

    es.indices.put_settings(index=index_name_combined,
                            body={'index': { 'refresh_interval': '1s',
                                            'number_of_replicas': '1',
                            }})

    # add removed articles from v2, which are referenced in the WAPO judgement list
    with open(f"{data_location}/wapo_missing_articles.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            article_id = line.strip()
            missing_article_es = es.get(index=index_name_v2, id=article_id)
            es.index(index=index_name_combined, id=missing_article_es["_id"], body=missing_article_es["_source"])