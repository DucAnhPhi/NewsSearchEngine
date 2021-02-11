from elasticsearch import Elasticsearch
from .parser import ParserNetzpolitik
from ..embedding.model import EmbeddingModel
from ..feature_extraction import FeatureExtraction
from ..vector_storage import VectorStorage
import os

if __name__ == "__main__":
    es = Elasticsearch()
    parser = ParserNetzpolitik(es)
    em = EmbeddingModel()
    fe = FeatureExtraction(em, parser)
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    articles_path = f"{data_location}/netzpolitik.jsonl"

    print("Initialize vector storage of embeddings of titles w/ first paragraph.\n")
    def embedding_func_twfp(raw):
        return fe.get_first_paragraph_with_titles_embedding(raw)
    vs_twfp = VectorStorage()
    vs_twfp.add_items_from_file(articles_path, embedding_func_twfp)
    vs_twfp.save(f"{data_location}/netzpolitik_vs_twfp.bin")

    print("Initialize vector storage of embeddings of pre-annotated keywords.\n")
    def embedding_func_annotated_keywords(raw):
        return fe.get_keywords_embedding(raw)
    vs_annotated_k = VectorStorage()
    vs_annotated_k.add_items_from_file(articles_path, embedding_func_annotated_keywords)
    vs_annotated_k.save(f"{data_location}/netzpolitik_vs_annotated_k.bin")

    print("Initialize vector storage of embeddings of extracted tf-idf keywords.\n")
    def embedding_func_tf_idf_keywords(raw):
        raw["keywords"] = parser.get_keywords_tf_idf("netzpolitik", raw["id"])
        return fe.get_keywords_embedding(raw)
    vs_extracted_k = VectorStorage()
    vs_extracted_k.add_items_from_file(articles_path, embedding_func_tf_idf_keywords)
    vs_extracted_k.save(f"{data_location}/netzpolitik_vs_extracted_k.bin")

    print("Initialize vector storage of embeddings of titles and section titles.\n")
    def embedding_func_section_titles(raw):
        return fe.get_titles_w_section_titles_embedding(raw)
    vs_section_titles = VectorStorage()
    vs_section_titles.add_items_from_file(articles_path, embedding_func_section_titles)
    vs_section_titles.save(f"{data_location}/netzpolitik_vs_section_titles.bin")

    print("Initialize vector storage of embeddings of titles.\n")
    def embedding_func_titles(raw):
        return fe.get_titles_embedding(raw)
    vs_titles = VectorStorage()
    vs_titles.add_items_from_file(articles_path, embedding_func_titles)
    vs_titles.save(f"{data_location}/netzpolitik_vs_titles.bin")