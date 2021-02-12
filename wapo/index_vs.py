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
    em = EmbeddingModel(lang="en")
    fe = FeatureExtraction(em, parser)
    index = "wapo_clean"
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    articles_path = f"{data_location}/TREC_Washington_Post_collection.v3.jl"

    print("Initialize WAPO vector storage of embeddings of titles w/ first paragraph.\n")
    def embedding_func_twfp(raw):
        article = parser.parse_article(raw)
        return fe.get_first_paragraph_with_titles_embedding(article)
    vs_twfp = VectorStorage(num_elements=500000)
    vs_twfp.add_items_from_file(articles_path, embedding_func_twfp)
    vs_twfp.save(f"{data_location}/wapo_vs_twfp.bin")

    print("Initialize WAPO vector storage of embeddings of keywords.\n")
    def embedding_func_keywords(raw):
        article = parser.parse_article(raw)
        combined = [article["title"], article["text"]]
        combined_text = " ".join([t for t in combined if t])
        article["keywords"] = parser.get_keywords_tf_idf_denormalized(index, article_id, combined_text)
        #article["keywords"] = parser.get_keywords_tf_idf(index, article_id)
        return fe.get_keywords_embedding(article)
    vs_keywords = VectorStorage(num_elements=500000)
    vs_keywords.add_items_from_file(articles_path, embedding_func_keywords)
    vs_keywords.save(f"{data_location}/wapo_vs_keywords.bin")