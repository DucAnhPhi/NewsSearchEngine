import os
import json
from elasticsearch import Elasticsearch
from ...vector_storage import VectorStorage
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...typings import NearestNeighborList

if __name__ == "__main__":
    # Semantic search titles recall avg: 0.035107300468283476
    es = Elasticsearch()
    index = "netzpolitik"
    fe = FeatureExtraction(EmbeddingModel(), ParserNetzpolitik())
    count = 0
    recall_avg = 0.
    # init vector storage
    storage_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}"
    if os.path.isfile(storage_location):
        print("Loading vector storage from file...\n")
        vs = VectorStorage(f"{storage_location}/data/storage_titles.bin")
    else:
        print("Initialize vector storage.\n")
        vs = VectorStorage()
        print("Add items from file...\n")
        vs.add_items_from_file(f"{storage_location}/data/netzpolitik.jsonl", fe.get_titles_embedding)
        vs.save(storage_location)
    # build query
    with open(f"{storage_location}/data/judgement_list_netzpolitik.jsonl", "r") as f:
        for line_i, line in enumerate(f):
            judgment = json.loads(line)
            try:
                count += 1
                query_article = (es.get(
                    index=index,
                    id=judgment["id"]
                ))["_source"]
                query = fe.get_titles_embedding(query_article)
                nearest_n: NearestNeighborList = vs.get_k_nearest([query],100)
                result_ids = [list(nn.keys())[0] for nn in nearest_n[0]]
                recall = 0.
                for res_id in result_ids:
                    print(res_id)
                    if res_id in query_article["references"]:
                        recall += 1
                recall /= len(query_article["references"])
                recall_avg += recall
            except:
                # cannot find query article
                count -= 1
                continue
    recall_avg /= count
    print(f"Semantic search titles recall avg: {recall_avg}")