import os.path
import pathlib
import json
from elasticsearch import Elasticsearch
from ...vector_storage import VectorStorage
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...typings import NearestNeighborList

if __name__ == "__main__":
    # Semantic search recall avg: 0.04029672880705244
    es = Elasticsearch()
    index = "netzpolitik"
    fe = FeatureExtraction(EmbeddingModel(), ParserNetzpolitik())
    count = 0
    recall_avg = 0.
    # init vector storage
    storage_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}"
    if os.path.isfile(storage_location):
        print("Loading vector storage from file...\n")
        vs = VectorStorage(storage_location)
    else:
        print("Initialize vector storage.\n")
        vs = VectorStorage()
        print("Add items from file...\n")
        vs.add_items_from_file(f"{storage_location}/data/netzpolitik.jsonl", fe.get_first_paragraph_with_titles_embedding)
        vs.save(f"{storage_location}/data/storage_titles_w_first_p.bin")
    # build query
    with open("data/judgement_list_netzpolitik.jsonl", "r") as f:
        for line_i, line in enumerate(f):
            judgment = json.loads(line)
            try:
                count += 1
                query_article = (es.get(
                    index=index,
                    id=judgment["id"]
                ))["_source"]
                query = fe.get_first_paragraph_with_titles_embedding(query_article)
                nearest_n: NearestNeighborList = vs.get_k_nearest([query],100)
                result_ids = [list(nn.keys())[0] for nn in nearest_n[0]]
                print(result_ids)
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
    print(f"Semantic search titles with first paragraph recall avg: {recall_avg}")