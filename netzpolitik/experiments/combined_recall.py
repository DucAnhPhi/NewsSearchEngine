from .keyword_match_recall import KeywordsMatchExperiment
from .semantic_search_recall import SemanticSearchExperiment
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...vector_storage import VectorStorage
from ...typings import NearestNeighborList, StringList
from elasticsearch import Elasticsearch
import os
import json

# Combined Semantic + Keyword Retrieval
# Recall Avg: 0.143914
# Retrieval Count Avg: 106.98838

if __name__ == "__main__":
    em = EmbeddingModel()
    fe = FeatureExtraction(em, ParserNetzpolitik())
    es = Elasticsearch()
    index = "netzpolitik"
    count = 0
    recall_avg = 0.
    retrieval_count_avg = 0.

    storage_file = "storage_titles_w_first_p.bin"

    # init vector storage
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    storage_location = f"{data_location}/{storage_file}"
    if os.path.isfile(storage_location):
        print("Loading vector storage from file...\n")
        vs = VectorStorage(storage_location)
    else:
        print("Initialize vector storage.\n")
        vs = VectorStorage()
        print("Add items from file...\n")
        vs.add_items_from_file(f"{data_location}/netzpolitik.jsonl", fe.get_first_paragraph_with_titles_embedding)
        vs.save(storage_location)

    # build query
    with open(f"{data_location}/judgement_list_netzpolitik.jsonl", "r") as f:
        for line in f:
            judgment = json.loads(line)
            try:
                count += 1
                query_article = (es.get(
                    index=index,
                    id=judgment["id"]
                ))["_source"]
                emb_query = fe.get_first_paragraph_with_titles_embedding(query_article)
                keyword_query = " ".join(query_article["keywords"])
                combined_result_ids: StringList = []
                if emb_query != None:
                    nearest_n: NearestNeighborList = vs.get_k_nearest([emb_query],100)
                    combined_result_ids = combined_result_ids + [list(nn.keys())[0] for nn in nearest_n[0]]
                if len(keyword_query) > 0:
                    results = es.search(
                        index = index,
                        body = {
                            "query": {
                                "multi_match": {
                                    "fields": [ "title", "subtitle", "body" ],
                                    "query": keyword_query
                                }
                            }
                        }
                    )
                    combined_result_ids = combined_result_ids + [res["_id"] for res in results["hits"]["hits"]]
                if len(combined_result_ids) == 0:
                    count -= 1
                    continue
                retrieval_count_avg += len(set(combined_result_ids))
                recall = 0.
                for res_id in set(combined_result_ids):
                    if res_id in query_article["references"]:
                        recall += 1/len(query_article["references"])
                recall_avg += recall
            except:
                # cannot find query article
                count -= 1
                continue
        recall_avg /= count
        retrieval_count_avg /= count
    print("----------------------------------------------------------------")
    print("Combined Retrieval")
    print(f"Recall Avg: {recall_avg}")
    print(f"Retrieval Count Avg: {retrieval_count_avg}")
    print("----------------------------------------------------------------\n")