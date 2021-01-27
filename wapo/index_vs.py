from ...wapo.parser import ParserWAPO
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...vector_storage import VectorStorage
import os
import json

if __name__ == "__main__":
    em = EmbeddingModel(lang="en")
    fe = FeatureExtraction(em, ParserWAPO())

    # init vector storage
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    storage_location_twfp = f"{data_location}/wapo_vs_twfp.bin"
    storage_location_keywords = f"{data_location}/wapo_vs_keywords.bin"

    print("Initialize twfp vector storage.\n")
    vs_twfp = VectorStorage()
    print("Add items from file...\n")
    vs_twfp.add_items_from_file(f"{data_location}/TREC_Washington_Post_collection.v3.jl", fe.get_first_paragraph_with_titles_embedding)
    vs_twfp.save(storage_location_twfp)

    print("Initialize keywords vector storage.\n")
    vs_keywords = VectorStorage()
    print("Add items from file...\n")
    vs_keywords.add_items_from_file(f"{data_location}/TREC_Washington_Post_collection.v3.jl", fe.get_keywords_embedding)
    vs_keywords.save(storage_location_keywords)