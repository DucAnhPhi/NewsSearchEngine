from .vector_storage import VectorStorage
import pprint
import os.path
from .netzpolitik.parser import ParserNetzpolitik
from .feature_extraction import FeatureExtraction
from .embedding.model import EmbeddingModel
import pathlib


if __name__ == "__main__":
    fe = FeatureExtraction(EmbeddingModel(), ParserNetzpolitik())
    pp = pprint.PrettyPrinter(indent=4)
    storage_location = f"{pathlib.Path(__file__).parent.absolute()}/data/storage.bin"
    if os.path.isfile(storage_location):
        print("Loading vector storage from file...\n")
        vs = VectorStorage(storage_location)
    else:
        print("Initialize vector storage.\n")
        vs = VectorStorage()
        print("Add items from file...\n")
        vs.add_items_from_file(f"{pathlib.Path(__file__).parent.absolute()}/data/netzpolitik.jsonl", fe.get_embedding_of_title_with_first_paragraph)
        vs.save(storage_location)
    while True:
        data = input("Get news based on your text: \n")
        recs = vs.get_k_nearest([data], 5)
        print("-----------------------------------------")
        print("Your recommendations: \n")
        for rec in recs:
            pp.pprint(rec)
        print("-----------------------------------------")