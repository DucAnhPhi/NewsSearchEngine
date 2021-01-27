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
from ..wapo.parser import ParserWAPO
from ..embedding.model import EmbeddingModel
from ..feature_extraction import FeatureExtraction

if __name__ == "__main__":
    embedder = EmbeddingModel(lang="en")
    parser = ParserWAPO()
    fe = FeatureExtraction(embedder, parser)
    parser = argparse.ArgumentParser(description='Index WashingtonPost docs to ElasticSearch')
    parser.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    parser.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    parser.add_argument('--index_name', default='wapo', help='index name')
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

    def get_first_content_by_type(jsarr, t):
        for block in jsarr:
            if block is not None and block['type'] == t:
                return block['content']
        return None

    def get_all_content_by_type(jsarr, t, field='content'):
        strings = [c[field] for c in jsarr if c is not None and c['type'] == t and field in c and c[field] is not None]
        if strings:
            return ' '.join(strings)
        else:
            return None

    def get_all_content_by_type_arr(jsarr, t, field="content"):
        strings = [c[field] for c in jsarr if c is not None and c['type'] == t and field in c and c[field] is not None]
        return strings

    def unique_heads(entry):
        items = set()
        if type(entry) is list:
            for x in entry:
                items.add(x[0])
            return list(items)
        else:
            return entry

    def is_not_relevant(kicker: str):
        not_relevant = {
            "test",
            "opinion",
            "letters to the editor",
            "the post's view"
        }
        return (kicker.lower() in not_relevant)

# TODO
    def get_keywords(text:str):
        keywords = []
        return keywords


    def doc_generator(f, num_docs):
        for line in tqdm(f, total=num_docs):
            js = json.loads(line)
            try:
                text_arr = get_all_content_by_type_arr(js['contents'], 'sanitized_html')
                first_p = text_arr[0] if len(text_arr) > 0 else None
                text = " ".join(text_arr) if text_arr else None
                links = []
                if text:
                    links = re.findall('href="([^"]*)"', text)
                    text = re.sub('<.*?>', ' ', text)
                title = get_all_content_by_type(js['contents'], 'title')
                kicker = get_first_content_by_type(js['contents'], 'kicker')

                # ignore not relevant docs
                if "published_date" not in js or not title.strip() or not text.strip() or is_not_relevant(kicker):
                    continue

                data_dict = {
                    "_index": args.index_name,
                    "_type": '_doc',
                    "_id": js['id'],
                }

                keywords = get_keywords(text)

                source_block = {
                    "title": title,
                    "offset_first_paragraph": len(first_p),
                    "date": js['published_date'],
                    "kicker": kicker,
                    "author": js['author'],
                    "text": text or '',
                    "links": links or [],
                    "url": js['article_url'],
                    "keywords": keywords,
                    "keywords_similarity": fe.get_keywords_similarity(keywords)

                }

                for key, val in js.items():
                    if key == key.upper():
                        print(js['id'], unique_heads(val))
                        source_block[key] = unique_heads(val)

                data_dict['_source'] = source_block

            except Exception:
                # print(json.dumps(js,sort_keys=True, indent=4))
                traceback.print_exc(file=sys.stdout)
                quit()

            yield data_dict

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
