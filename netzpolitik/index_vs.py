from elasticsearch import Elasticsearch
from .parser import ParserNetzpolitik
from ..embedding.model import EmbeddingModel
from ..feature_extraction import FeatureExtraction
from ..vector_storage import VectorStorage
import os

if __name__ == "__main__":
    es = Elasticsearch()
    parser = ParserNetzpolitik(es)
    lang = "de"
    em = EmbeddingModel(lang)
    fe = FeatureExtraction(em, parser)
    index = "netzpolitik"
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    articles_path = f"{data_location}/netzpolitik.jsonl"
    num_elements = 20000

    print("Initialize netzpolitik vector storage of embeddings of title.\n")
    def embedding_func_title(raw):
        return fe.get_embedding_of_title(raw)
    VectorStorage(f"{data_location}/netzpolitik_vs_title.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title)

    print("Initialize netzpolitik vector storage of embeddings of title and section titles.\n")
    def embedding_func_title_with_section_titles(raw):
        return fe.get_embedding_of_title_with_section_titles(raw)
    VectorStorage(f"{data_location}/netzpolitik_vs_title_with_section_titles.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title_with_section_titles)

    print("Initialize netzpolitik vector storage of embeddings of title with first paragraph.\n")
    def embedding_func_title_with_first_paragraph(raw):
        return fe.get_embedding_of_title_with_first_paragraph(raw)
    VectorStorage(f"{data_location}/netzpolitik_vs_title_with_first_paragraph.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_title_with_first_paragraph)

    print("Initialize netzpolitik vector storage of embeddings of pre-annotated keywords.\n")
    def embedding_func_annotated_keywords(raw):
        return fe.get_embedding_of_keywords(raw)
    VectorStorage(f"{data_location}/netzpolitik_vs_annotated_k.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_annotated_keywords)

    print("Initialize netzpolitik vector storage of embeddings of extracted tf-idf keywords (normalized, unordered).\n")
    def embedding_func_tf_idf_keywords(raw):
        keyw = parser.get_keywords_tf_idf(index, raw["id"])
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/netzpolitik_vs_extracted_k_normalized.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords)

    print("Initialize netzpolitik vector storage of embeddings of extracted tf-idf keywords (denormalized, unordered).\n")
    def embedding_func_tf_idf_keywords_denormalized(raw):
        keyw = parser.get_keywords_tf_idf_denormalized(index, raw["id"], raw["body"], keep_order=False)
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/netzpolitik_vs_extracted_k_denormalized.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords_denormalized)

    print("Initialize netzpolitik vector storage of embeddings of extracted tf-idf keywords (denormalized, order preserved).\n")
    def embedding_func_tf_idf_keywords_denormalized_ordered(raw):
        keyw = parser.get_keywords_tf_idf_denormalized(index, raw["id"], raw["body"], keep_order=True)
        return fe.get_embedding_of_keywords(keyw)
    VectorStorage(f"{data_location}/netzpolitik_vs_extracted_k_denormalized_ordered.bin", num_elements) \
        .add_items_from_file(articles_path, embedding_func_tf_idf_keywords_denormalized_ordered)