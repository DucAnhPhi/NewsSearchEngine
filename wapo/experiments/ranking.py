import argparse
import os
import lightgbm as lgb
import numpy as np
import json
from tqdm import tqdm
from elasticsearch import Elasticsearch
from scipy.spatial.distance import cosine
from ..parser import ParserWAPO
from ...embedding.model import EmbeddingModel
from ...feature_extraction import FeatureExtraction
from ..judgement_list import JudgementListWapo
from ...vector_storage import VectorStorage

def get_embedding_of_extracted_keywords_denormalized_ordered(parser, index, es_doc):
        keywords = parser.get_keywords_tf_idf_denormalized(index, es_doc["_id"], es_doc["_source"]["title"], es_doc["_source"]["text"], keep_order=True)
        query = " ".join(keywords)
        return em.encode(query)

def get_features(es, parser, index, query_es, doc_id, bm25_score=None, cosine_score=None):
    doc_es = es.get(index=index, id=doc_id)

    doc_length = len(doc_es["_source"]["title"]) + len(doc_es["_source"]["text"])
    query_published_after = 1 if int(query_es["_source"]["date"]) > int(doc_es["_source"]["date"]) else 0

    if not bm25_score:
        bm25_score = 0
        query_keywords = parser.get_keywords_tf_idf(index, query_es["_id"])
        query = " OR ".join(query_keywords)
        if query:
            val = es.explain(
                index=index,
                id=doc_id,
                body = {
                    "query": {
                        "query_string": {
                            "fields": [ "title", "text" ],
                            "query": query
                        }
                    }
                }
            )["explanation"]["value"]
            if val:
                bm25_score = val
    if not cosine_score:
        cosine_score = 1
        query_emb = get_embedding_of_extracted_keywords_denormalized_ordered(parser, index, query_es)
        doc_emb = get_embedding_of_extracted_keywords_denormalized_ordered(parser, index, doc_es)
        if query_emb and doc_emb:
            cosine_score = cosine(query_emb, doc_emb)
    
    return [bm25_score, cosine_score, doc_length, query_published_after]

def get_training_and_validation_data(es, parser, index):
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    judgement_list_paths = [f"{data_location}/judgement_list_wapo_18.jsonl", f"{data_location}/judgement_list_wapo_19.jsonl"]
    data = []
    for path in judgement_list_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                data.append(judgement)
    np.random.shuffle(data)
    train_size = int(len(data) * 0.7)
    train_data_raw, val_data_raw = data[:train_size], data[train_size:]
    X_train = []
    y_train = []
    query_train = []
    for jl in tqdm(train_data_raw, total=77):
        query_es = es.get(index=index, id=jl["id"])
        query_train.append(len(jl["references"]))
        for ref in jl["references"]:
            ref_features = get_features(es, parser, index, query_es, ref["id"])
            X_train.append(ref_features)
            y_train.append(int(ref["exp_rel"]))
    X_val = []
    y_val = []
    query_val = []
    for jl in tqdm(val_data_raw, total=33):
        query_es = es.get(index=index, id=jl["id"])
        query_val.append(len(jl["references"]))
        for ref in jl["references"]:
            ref_features = get_features(es, parser, index, query_es, ref["id"])
            X_val.append(ref_features)
            y_val.append(int(ref["exp_rel"]))
    return (np.array(X_train), np.array(y_train), np.array(query_train), np.array(X_val), np.array(y_val), np.array(query_val))

def get_combined_retrieval(es, parser, em, vs, index, size, query_id):
    result_ids = []
    query_article_es = es.get(
        index=index,
        id=query_id
    )
    keywords = parser.get_keywords_tf_idf(index, query_id)
    if keywords:
        query_keywords = " OR ".join(keywords)
        k_results = (es.search(
            size = size,
            index = index,
            body = {
                "query": {
                    "query_string": {
                        "fields": [ "title", "text" ],
                        "query": query_keywords
                    }
                }
            }
        ))["hits"]["hits"]
        result_ids = [{"id": res["_id"], "bm25_score":res["_score"], "cosine_score":None} for res in k_results]
    keywords = parser.get_keywords_tf_idf_denormalized(index, query_article_es["_id"], query_article_es["_source"]["title"], query_article_es["_source"]["text"], keep_order=True)
    keyw_query = " ".join(keywords)
    if not keyw_query:
        return None
    emb_query = em.encode(keyw_query)
    if emb_query:
        nearest_n: NearestNeighborList = vs.get_k_nearest(emb_query,size)
        e_results = [{"id": list(nn.keys())[0], "cosine_score":list(nn.values())[0], "bm25_score":None} for nn in nearest_n[0]]
        for res_e in e_results:
            for i, res_k in enumerate(result_ids):
                if res_e["id"] == res_k["id"]:
                    result_ids[i]["cosine_score"] = res_e["cosine_score"]
                    break
                result_ids.append(res_e)
    return result_ids

def test_model(es, parser, em, vs, index, size, judgement_list_path, result_path, model):
    X_test = []
    query_test = []
    topic_dict = {v: k for k, v in (JudgementListWapo.get_topic_dict("20")).items()}
    topic_ids = []
    X_test_ids = []
    print("Retrieve background links for each topic and calculate features...")
    with open(judgement_list_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=50):
            judgement = json.loads(line)
            topic_ids.append(topic_dict[judgement["id"]])
            combined_retrieval = get_combined_retrieval(es, parser, em, vs, index, size, judgement["id"])
            query_test.append(len(combined_retrieval))
            for res in combined_retrieval:
                doc_features = get_features(es,parser,index,query_es,res["id"],res["bm25_score"],res["cosine_score"])
                X_test_ids.append(res["id"])
                X_test.append(doc_features)
    print("Rank features with trained ranking model and output results to file...")
    test_pred = model.predict(X_test)
    acc_offset = 0
    with open(result_path, "w", encoding="utf-8") as fout:
        for i, offset in enumerate(query_test):
            test_pred_topic = np.array(test_pred[acc_offset:(acc_offset+offset)])
            inds = (test_pred_topic.argsort())[::-1]
            ranked_test_pred = test_pred_topic[inds]
            ranked_retrieval = (X_test_ids[acc_offset:(acc_offset+offset)])[inds]
            topic = topic_ids[i]
            for rank, ret in enumerate(ranked_retrieval):
                out = f"{topic}\tQ0\t{ret}\t{rank}\t{ranked_test_pred[rank]}\tducrun\n"
                fout.write(out)
            acc_offset += offset
    print("Finished.")

if __name__ == "__main__":
    index = "wapo_clean"

    p = argparse.ArgumentParser(description='Run Washington Post semantic search retrieval experiments')
    p.add_argument('--host', default='localhost', help='Host for ElasticSearch endpoint')
    p.add_argument('--port', default='9200', help='Port for ElasticSearch endpoint')
    p.add_argument('--index_name', default=index, help='index name')
    p.add_argument('--user', default=None, help='ElasticSearch user')
    p.add_argument('--secret', default=None, help="ElasticSearch secret")
    p.add_argument('--device', default="cpu", help="(CUDA) device for pytorch")

    args = p.parse_args()

    es = None

    if args.user and args.secret:
        es = Elasticsearch(
            hosts = [{"host": args.host, "port": args.port}],
            http_auth=(args.user, args.secret),
            scheme="https",
            retry_on_timeout=True,
            max_retries=10
        )
    else:
        es = Elasticsearch(
            hosts=[{"host": args.host, "port": args.port}],
            retry_on_timeout=True,
            max_retries=10
        )

    parser = ParserWAPO(es)
    em = EmbeddingModel(lang="en", device=args.device)
    fe = FeatureExtraction(em, parser)
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    judgement_list_20_path = f"{data_location}/judgement_list_wapo_20.jsonl"
    vs_extracted_k_denormalized_ordered = f"{data_location}/wapo_vs_extracted_k_denormalized_ordered.bin"
    vs = VectorStorage(vs_extracted_k_denormalized_ordered)

    if not os.path.isfile(f"{data_location}/ranking_model.txt"):
        if not os.path.isfile(f"{data_location}/X_train.txt"):
            print("Initialize training and validation data...")
            X_train, y_train, query_train, X_val, y_val, query_val = get_training_and_validation_data(es, parser, index)

            print("Finished. Saving data...")
            np.savetxt(f"{data_location}/X_train.txt", X_train)
            np.savetxt(f"{data_location}/y_train.txt", y_train)
            np.savetxt(f"{data_location}/X_val.txt", X_val)
            np.savetxt(f"{data_location}/y_val.txt", y_val)
            np.savetxt(f"{data_location}/query_train.txt", query_train)
            np.savetxt(f"{data_location}/query_val.txt", query_val)
        else:
            X_train = np.loadtxt(f"{data_location}/X_train.txt", dtype=float)
            y_train = np.loadtxt(f"{data_location}/y_train.txt", dtype=float)
            query_train = np.loadtxt(f"{data_location}/query_train.txt", dtype=float)
            X_val = np.loadtxt(f"{data_location}/X_val.txt", dtype=float)
            y_val = np.loadtxt(f"{data_location}/y_val.txt", dtype=float)
            query_val = np.loadtxt(f"{data_location}/query_val.txt", dtype=float)

        gbm = lgb.LGBMRanker()
        print("Start training...")
        gbm.fit(
            X_train,
            y_train,
            group=query_train,
            eval_set=[(X_val, y_val)],
            eval_group=[query_val],
            eval_at=[5,10],
            early_stopping_rounds=50
        )
        print("Training finished. Saving ranking model...")
        gbm.booster_.save_model(f"{data_location}/ranking_model.txt")

    if os.path.isfile(f"{data_location}/ranking_model.txt"):
        print("Load model from file...")
        result_path = f"{data_location}/ranking_results_20.txt"
        test_model(es, parser, em, vs, index, 300, judgement_list_20_path, result_path, gbm)