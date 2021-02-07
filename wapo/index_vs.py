from .parser import ParserWAPO
from ..feature_extraction import FeatureExtraction
from ..embedding.model import EmbeddingModel
from ..vector_storage import VectorStorage
from elasticsearch import Elasticsearch
from ..typings import StringList, VectorList
import os
import json

if __name__ == "__main__":
    em = EmbeddingModel(lang="en")
    es = Elasticsearch()
    parser = ParserWAPO(es)
    fe = FeatureExtraction(em, parser)
    index = "wapo_clean"

    # init vector storage
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir))}/data"
    storage_location_twfp = f"{data_location}/wapo_vs_twfp.bin"
    storage_location_keywords = f"{data_location}/wapo_vs_keywords.bin"

    print("Initialize twfp vector storage.\n")
    vs_twfp = VectorStorage()
    print("Add items from file...\n")
    with open(f"{data_location}/TREC_Washington_Post_collection.v3.jl", 'r') as data_file:
        emb_batch: VectorList = []
        id_batch: StringList = []

        for line in data_file:
            article = json.loads(line)
            article = parser.parse_article(article, index)
            emb = fe.get_first_paragraph_with_titles_embedding(article)
            if emb == None:
                continue
            emb_batch.append(emb)
            id_batch.append(article["_id"])

            if len(emb_batch) is 1000:
                vs_twfp.add_items(emb_batch, id_batch)
                emb_batch = []
                id_batch = []

        if len(emb_batch) is not 0:
            vs_twfp.add_items(emb_batch, id_batch)

    vs_twfp.save(storage_location_twfp)

    print("Initialize keywords vector storage.\n")
    vs_keywords = VectorStorage()
    print("Add items from file...\n")
    with open(f"{data_location}/TREC_Washington_Post_collection.v3.jl", 'r') as data_file:
        emb_batch_key: VectorList = []
        id_batch_key: StringList = []

        for line in data_file:
            article = json.loads(line)
            article = parser.parse_article(article, index)
            article["keywords"] = parser.get_keywords(article)
            emb = fe.get_keywords_embedding(article)
            if emb == None:
                continue
            emb_batch_key.append(emb)
            id_batch_key.append(article["_id"])

            if len(emb_batch_key) is 1000:
                vs_keywords.add_items(emb_batch_key, id_batch_key)
                emb_batch_key = []
                id_batch_key = []

        if len(emb_batch_key) is not 0:
            vs_keywords.add_items(emb_batch_key, id_batch_key)
    vs_keywords.save(storage_location_keywords)