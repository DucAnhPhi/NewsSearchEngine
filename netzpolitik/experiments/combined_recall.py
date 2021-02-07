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


    # init vector storage
    storage_file_twfp = "storage_titles_w_first_p.bin"
    storage_file_keywords = "storage_keywords.bin"
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    storage_location_twfp = f"{data_location}/{storage_file_twfp}"
    storage_location_keywords = f"{data_location}/{storage_file_keywords}"

    if os.path.isfile(storage_location_twfp):
        print("Loading twfp vector storage from file...\n")
        vs_twfp = VectorStorage(storage_location_twfp)
    else:
        print("Initialize twfp vector storage.\n")
        vs_twfp = VectorStorage()
        print("Add items from file...\n")
        with open(f"{data_location}/netzpolitik.jsonl", 'r') as data_file:
            emb_batch = []
            id_batch: StringList = []

            for line in data_file:
                article = json.loads(line)
                emb = fe.get_first_paragraph_with_titles_embedding(article)
                if emb == None:
                    continue
                emb_batch.append(emb)
                id_batch.append(article["id"])

                if len(emb_batch) is 1000:
                    vs_twfp.add_items(emb_batch, id_batch)
                    emb_batch = []
                    id_batch = []

            if len(emb_batch) is not 0:
                vs_twfp.add_items(emb_batch, id_batch)
        vs_twfp.save(storage_location_twfp)

    if os.path.isfile(storage_location_keywords):
        print("Loading keywords vector storage from file...\n")
        vs_keywords = VectorStorage(storage_location_keywords)
    else:
        print("Initialize keywords vector storage.\n")
        vs_keywords = VectorStorage()
        print("Add items from file...\n")
        with open(f"{data_location}/netzpolitik.jsonl", 'r') as data_file:
            emb_batch_key = []
            id_batch_key: StringList = []

            for line in data_file:
                article = json.loads(line)
                emb = fe.get_keywords_embedding(article)
                if emb == None:
                    continue
                emb_batch_key.append(emb)
                id_batch_key.append(article["id"])

                if len(emb_batch_key) is 1000:
                    vs_keywords.add_items(emb_batch_key, id_batch_key)
                    emb_batch_key = []
                    id_batch_key = []

            if len(emb_batch_key) is not 0:
                vs_keywords.add_items(emb_batch_key, id_batch_key)
        vs_keywords.save(storage_location_keywords)

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
                twfp_emb_query = fe.get_first_paragraph_with_titles_embedding(query_article)
                keyword_emb_query = fe.get_keywords_embedding(query_article)
                keyword_match_query = " ".join(query_article["keywords"])
                combined_result_ids: StringList = []
                if twfp_emb_query != None:
                    nearest_n_twfp: NearestNeighborList = vs_twfp.get_k_nearest([twfp_emb_query],100)
                    combined_result_ids = combined_result_ids + [list(nn.keys())[0] for nn in nearest_n_twfp[0]]
                if keyword_emb_query != None:
                    nearest_n_keywords: NearestNeighborList = vs_keywords.get_k_nearest([keyword_emb_query], 100)
                    combined_result_ids = combined_result_ids + [list(nn.keys())[0] for nn in nearest_n_keywords[0]]
                if len(keyword_match_query) > 0:
                    results = es.search(
                        index = index,
                        body = {
                            "query": {
                                "multi_match": {
                                    "fields": [ "title", "subtitle", "body" ],
                                    "query": keyword_match_query
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