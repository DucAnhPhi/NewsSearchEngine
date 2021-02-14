from .parser import ParserWAPO
from ..feature_extraction import FeatureExtraction
from ..embedding.model import EmbeddingModel
from ..vector_storage import VectorStorage
from elasticsearch import Elasticsearch
from ..typings import StringList, VectorList
import os
import json

if __name__ == "__main__":
    es = Elasticsearch()
    parser = ParserWAPO(es)
    lang = "en"
    em = EmbeddingModel(lang)
    fe = FeatureExtraction(em, parser)
    index = "wapo_clean"
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    articles_path = f"{data_location}/TREC_Washington_Post_collection.v3.jl"
    num_elements = 500000

    print("Initialize WAPO vector storage of embeddings of title.\n")
    def embedding_func_title(raw):
        article = parser.parse_article(raw)
        return fe.get_embedding_of_title(article)
    VectorStorage(f"{data_location}/wapo_vs_title.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title)

    print("Initialize WAPO vector storage of embeddings of title and section titles.\n")
    def embedding_func_title_with_section_titles(raw):
        article = parser.parse_article(raw)
        return fe.get_embedding_of_title_with_section_titles(article)
    VectorStorage(f"{data_location}/wapo_vs_title_with_section_titles.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title_with_section_titles)

    print("Initialize WAPO vector storage of embeddings of title with first paragraph.\n")
    def embedding_func_title_with_first_paragraph(raw):
        article = parser.parse_article(raw)
        return fe.get_embedding_of_title_with_first_paragraph(article)
    VectorStorage(f"{data_location}/wapo_vs_title_with_first_paragraph.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title_with_first_paragraph)

    print("Initialize WAPO vector storage of embeddings of extracted tf-idf keywords (normalized, unordered).\n")
    def embedding_func_tf_idf_keywords(raw):
        keyw = parser.get_keywords_tf_idf(index, raw["id"])
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/wapo_vs_extracted_k_normalized.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords)

    print("Initialize WAPO vector storage of embeddings of extracted tf-idf keywords (denormalized, unordered).\n")
    def embedding_func_tf_idf_keywords_denormalized(raw):
        article = parser.parse_article(raw)
        keyw = parser.get_keywords_tf_idf_denormalized(index, raw["id"], article["body"], keep_order=False)
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/wapo_vs_extracted_k_denormalized.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords_denormalized)

    print("Initialize WAPO vector storage of embeddings of extracted tf-idf keywords (denormalized, order preserved).\n")
    def embedding_func_tf_idf_keywords_denormalized_ordered(raw):
        article = parser.parse_article(raw)
        keyw = parser.get_keywords_tf_idf_denormalized(index, raw["id"], article["body"], keep_order=True)
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/wapo_vs_extracted_k_denormalized_ordered.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords_denormalized_ordered)