import os
import json
from elasticsearch import Elasticsearch
from ...vector_storage import VectorStorage
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...typings import NearestNeighborList

if __name__ == "__main__":
    # Semantic search keywords recall avg: 0.1107625078261979
    # Retrieval count avg: 100.58163127282565
    es = Elasticsearch()
    index = "netzpolitik"
    em = EmbeddingModel()
    fe = FeatureExtraction(em, ParserNetzpolitik())
    count = 0
    recall_avg = 0.
    retrieval_count_avg = 0.
    # init vector storage
    storage_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}"
    if os.path.isfile(f"{storage_location}/data/storage_keywords.bin"):
        print("Loading vector storage from file...\n")
        vs = VectorStorage(f"{storage_location}/data/storage_keywords.bin")
    else:
        print("Initialize vector storage.\n")
        vs = VectorStorage()
        print("Add items from file...\n")
        vs.add_items_from_file(f"{storage_location}/data/netzpolitik.jsonl", fe.get_keywords_embedding)
        vs.save(f"{storage_location}/data/storage_keywords.bin")
    # build query
    with open(f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data/judgement_list_netzpolitik.jsonl", "r") as f:
        for line in f:
            judgment = json.loads(line)
            try:
                query_article = (es.get(
                    index=index,
                    id=judgment["id"]
                ))["_source"]
                query = " ".join(query_article["keywords"])
                if len(query) == 0:
                    continue
                query_emb = em.encode(query)
                count += 1
                nearest_n: NearestNeighborList = vs.get_k_nearest([query_emb],100)
                result_ids = [list(nn.keys())[0] for nn in nearest_n[0]]
                recall = 0.
                retrieval_count_avg += len(result_ids)
                for res_id in result_ids:
                    if res_id in query_article["references"]:
                        recall += 1
                recall /= len(query_article["references"])
                recall_avg += recall
            except:
                # cannot find query article
                count -= 1
                continue
    recall_avg /= count
    retrieval_count_avg /= count
    print(f"Semantic search keywords recall avg: {recall_avg}")
    print(f"Retrieval count avg: {retrieval_count_avg}")