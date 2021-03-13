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
    index_name = "wapo_clean"

    p = argparse.ArgumentParser(description='Index Washington Post articles to ElasticSearch')
    p.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    p.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    p.add_argument('--index_name', default=index_name, help='index name')
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

    if not es.indices.exists(index=args.index_name):
        try:
            es.indices.create(index=args.index_name, body=settings)
        except TransportError as e:
            print(e.info)
            sys.exit(-1)

    def doc_generator(f, num_docs):
        for line in tqdm(f, total=num_docs):
            raw = json.loads(line)
            article_source = parser.parse_article(raw)
            if article_source != None:
                data_dict = {
                    "_index": index_name,
                    "_id": raw['id'],
                }
                data_dict["_source"] = article_source
                yield data_dict

    print("Counting...")
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    articles_location = f"{data_location}/TREC_Washington_Post_collection.v3.jl"
    with open(articles_location, 'r', encoding="utf-8") as f:
        lines = 0
        for line in f:
            lines += 1

    print("Indexing...")
    with open(articles_location, 'r', encoding="utf-8") as f:
        helpers.bulk(es, doc_generator(f, lines), request_timeout=30)

    es.indices.put_settings(index=args.index_name,
                            body={'index': { 'refresh_interval': '1s',
                                            'number_of_replicas': '1',
                            }})

    missing_test_article = {
        "title": "Congress betrays Afghans who risked their lives to help the U.S.",
        "date": 1463441555000,
        "kicker": "The Post's View",
        "author": ["Editorial Board"],
        "text": "THE HOUSE and Senate Amed Services committees have let down Afghans who braved danger and strife to work with the United States through a decade and a half of war. In 2009, Congress approved  a special program  that provided visas for Afghan translators and others who risked their lives shoulder-to-shoulder with U.S. forces. Though 8,528 applicants  have received visas  , thousands are still waiting. The annual defense bills just sent to the floor of each chamber could leave many of them behind. The legislation should be amended to keep faith with those who kept faith with the United States. The House  legislation  contains a provision that would limit eligibility for new applicants after May 31 to those who had served off-base as military translators or with the U.S. military in a “sensitive and trusted” role. This would exclude Afghans working on bases such as firefighters, maintenance workers and clerical assistants. When we asked Armed Services Committee Chairman Mac Thornberry (R-Tex.) about the limitations, he said they were a response to information he received that some Afghans were exploiting the program, taking menial jobs just to get a visa. We think the response is misplaced. Afghans who work with the United States, whether pushing a broom or translating under fire, potentially face the danger of retribution from extremists — especially when they go home at night. Mr. Thornberry’s concern is already dealt with in the visa process, which requires a detailed check to verify that applicants face ongoing danger. Last year, Congress lengthened the minimum service to two years. If the average processing time of 270 business days is counted, that means most are working for the United States for three years or more before they can acquire a visa and travel. Mr. Thornberry said, “If they can prove they are in danger, I am for them.” We urge him — and the House — to scrap the new limitations and leave the door open for every qualified applicant who faces danger because of association with the United States. The program also needs more visas. The State Department  reported   this year a backlog of approximately 10,300 principal applicants at some step in the process, with only about 4,000 visas remaining to be distributed. If Congress does not provide more, about 6,000 applicants will be left out in the cold, according to the  International Refugee Assistance Project . The House  bill  extends the program through 2017 but includes no new visas, while the Senate bill is silent on the program altogether. Both chambers need to act and allocate new visas. The special Afghan visa program has been a valuable tool for recruiting people to serve in a war zone. Cutting off thousands of applicants will not only betray their loyalty but also send the wrong signal, far and wide, about working for the United States.  Read more on this topic:   The Post’s View: Don’t endanger visas for the Afghans who helped U.S. troops   Aaron E. Fleming: My Afghan battle partner deserves a U.S. visa   Ryan Crocker: Don’t let the U.S. abandon thousands of Afghans who worked for us ",
        "url": "https://www.washingtonpost.com/opinions/congress-betrays-afghans-who-risked-their-lives-to-help-the-us/2016/05/16/516fb0da-1b92-11e6-9c81-4be1c14fb8c8_story.html",
        "section_titles": [],
        "offset_first_paragraph": 163,
        "links": [
            "https://travel.state.gov/content/visas/en/immigrate/afghans-work-for-us.html",
            "https://travel.state.gov/content/dam/visas/Statistics/Immigrant-Statistics/SIV/SQNumbers0316.pdf",
            "https://www.congress.gov/bill/114th-congress/house-bill/1735/text",
            "https://travel.state.gov/content/dam/visas/SIVs/Afghan%20SIV%20public%20report_Jan%202016.pdf",
            "http://www.refugeerights.org/",
            "https://www.congress.gov/bill/114th-congress/house-bill/1735/text",
            "https://www.washingtonpost.com/opinions/dont-endanger-visas-for-the-afghans-who-helped-us-troops/2016/02/05/971557c0-cb60-11e5-88ff-e2d1b4289c2f_story.html",
            "https://www.washingtonpost.com/opinions/my-afghan-battle-partner-deserves-a-us-visa/2015/11/27/98b00ab6-912d-11e5-a2d6-f57908580b1f_story.html",
            "https://www.washingtonpost.com/posteverything/wp/2016/05/12/dont-let-the-u-s-abandon-thousands-of-afghans-who-worked-for-us/"
        ]
    }
    es.index(index=index_name, id="516fb0da-1b92-11e6-9c81-4be1c14fb8c8", body=missing_test_article)