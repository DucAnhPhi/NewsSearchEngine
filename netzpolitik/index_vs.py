import os
import argparse
from elasticsearch import Elasticsearch
from .parser import ParserNetzpolitik
from ..embedding.model import EmbeddingModel
from ..feature_extraction import FeatureExtraction
from ..vector_storage import VectorStorage

if __name__ == "__main__":
    index = "netzpolitik"
    p = argparse.ArgumentParser(description='Index netzpolitik.org articles to vector storage')
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

    parser = ParserNetzpolitik(es)
    lang = "de"
    em = EmbeddingModel(lang, device=args.device)
    fe = FeatureExtraction(em, parser)
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    articles_path = f"{data_location}/netzpolitik.jsonl"
    num_elements = 20000

    def get_article_id(raw):
        indexed = (es.search(
            index=index,
            body={
                "query": {
                    "term": {
                        "url": {
                            "value": raw["url"]
                        }
                    }
                }
            }
        ))["hits"]["hits"]
        res = None
        if indexed and indexed[0]:
            res = indexed[0]["_id"]
        return res

    print("Initialize netzpolitik vector storage of embeddings of title.\n")
    def embedding_func_title(raw):
        return fe.get_embedding_of_title(raw)
    VectorStorage(f"{data_location}/netzpolitik_vs_title.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title, get_article_id)

    print("Initialize netzpolitik vector storage of embeddings of title and section titles.\n")
    def embedding_func_title_with_section_titles(raw):
        return fe.get_embedding_of_title_with_section_titles(raw)
    VectorStorage(f"{data_location}/netzpolitik_vs_title_with_section_titles.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title_with_section_titles, get_article_id)

    print("Initialize netzpolitik vector storage of embeddings of title with first paragraph.\n")
    def embedding_func_title_with_first_paragraph(raw):
        return fe.get_embedding_of_title_with_first_paragraph(raw)
    VectorStorage(f"{data_location}/netzpolitik_vs_title_with_first_paragraph.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title_with_first_paragraph, get_article_id)

    print("Initialize netzpolitik vector storage of embeddings of pre-annotated keywords.\n")
    def embedding_func_annotated_keywords(raw):
        return fe.get_embedding_of_keywords(raw["keywords"])
    VectorStorage(f"{data_location}/netzpolitik_vs_annotated_k.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_annotated_keywords, get_article_id)

    print("Initialize netzpolitik vector storage of embeddings of extracted tf-idf keywords (normalized, unordered).\n")
    def embedding_func_tf_idf_keywords(raw):
        article_id = get_article_id(raw)
        if not article_id:
            return None
        keyw = parser.get_keywords_tf_idf(args.index_name, article_id)
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/netzpolitik_vs_extracted_k_normalized.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords, get_article_id)

    print("Initialize netzpolitik vector storage of embeddings of extracted tf-idf keywords (denormalized, unordered).\n")
    def embedding_func_tf_idf_keywords_denormalized(raw):
        article_id = get_article_id(raw)
        if not article_id:
            return None
        keyw = parser.get_keywords_tf_idf_denormalized(args.index_name, article_id, raw["body"], keep_order=False)
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/netzpolitik_vs_extracted_k_denormalized.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords_denormalized, get_article_id)

    print("Initialize netzpolitik vector storage of embeddings of extracted tf-idf keywords (denormalized, order preserved).\n")
    def embedding_func_tf_idf_keywords_denormalized_ordered(raw):
        article_id = get_article_id(raw)
        if not article_id:
            return None
        keyw = parser.get_keywords_tf_idf_denormalized(args.index_name, article_id, raw["body"], keep_order=True)
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/netzpolitik_vs_extracted_k_denormalized_ordered.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords_denormalized_ordered, get_article_id)