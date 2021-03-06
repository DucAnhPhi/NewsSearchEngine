from .parser import ParserWAPO
from ..feature_extraction import FeatureExtraction
from ..embedding.model import EmbeddingModel
from ..vector_storage import VectorStorage
from elasticsearch import Elasticsearch
from ..typings import StringList, VectorList
import os
import argparse
import json

if __name__ == "__main__":
    index = "wapo_clean"

    p = argparse.ArgumentParser(description='Index Washington Post articles to vector storage')
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

    parser = ParserWAPO(es)
    lang = "en"
    em = EmbeddingModel(lang, device=args.device)
    fe = FeatureExtraction(em, parser)
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    articles_path = f"{data_location}/TREC_Washington_Post_collection.v3.jl"
    num_elements = 500000

    def get_article_id(raw):
        return raw["id"]

    print("Initialize WAPO vector storage of embeddings of title.\n")
    def embedding_func_title(raw):
        article = parser.parse_article(raw)
        return fe.get_embedding_of_title(article)
    VectorStorage(f"{data_location}/wapo_vs_title.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title, get_article_id)

    print("Initialize WAPO vector storage of embeddings of title and section titles.\n")
    def embedding_func_title_with_section_titles(raw):
        article = parser.parse_article(raw)
        return fe.get_embedding_of_title_with_section_titles(article)
    VectorStorage(f"{data_location}/wapo_vs_title_with_section_titles.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title_with_section_titles, get_article_id)

    print("Initialize WAPO vector storage of embeddings of title with first paragraph.\n")
    def embedding_func_title_with_first_paragraph(raw):
        article = parser.parse_article(raw)
        return fe.get_embedding_of_title_with_first_paragraph(article)
    VectorStorage(f"{data_location}/wapo_vs_title_with_first_paragraph.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title_with_first_paragraph, get_article_id)

    print("Initialize WAPO vector storage of embeddings of extracted tf-idf keywords (normalized, unordered).\n")
    def embedding_func_tf_idf_keywords(raw):
        article = parser.parse_article(raw)
        if article:
            keyw = parser.get_keywords_tf_idf(index, raw["id"])
            return fe.get_embedding_of_keywords(keyw)
        return None
    VectorStorage(f"{data_location}/wapo_vs_extracted_k_normalized.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords, get_article_id)

    print("Initialize WAPO vector storage of embeddings of extracted tf-idf keywords (denormalized, unordered).\n")
    def embedding_func_tf_idf_keywords_denormalized(raw):
        article = parser.parse_article(raw)
        if article:
            keyw = parser.get_keywords_tf_idf_denormalized(index, raw["id"], article["text"], keep_order=False)
            return fe.get_embedding_of_keywords(keyw)
        return None
    VectorStorage(f"{data_location}/wapo_vs_extracted_k_denormalized.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords_denormalized, get_article_id)

    print("Initialize WAPO vector storage of embeddings of extracted tf-idf keywords (denormalized, order preserved).\n")
    def embedding_func_tf_idf_keywords_denormalized_ordered(raw):
        article = parser.parse_article(raw)
        if article:
            keyw = parser.get_keywords_tf_idf_denormalized(index, raw["id"], article["text"], keep_order=True)
            return fe.get_embedding_of_keywords(keyw)
        return None
    VectorStorage(f"{data_location}/wapo_vs_extracted_k_denormalized_ordered.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords_denormalized_ordered, get_article_id)