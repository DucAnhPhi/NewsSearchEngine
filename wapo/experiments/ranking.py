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
from ...typings import NearestNeighborList

class WAPORanker():
    def __init__(self, es, parser, em, vs, index):
        self.es = es
        self.parser = parser
        self.em = em
        self.vs = vs
        self.index = index

    def get_embedding_of_extracted_keywords_denormalized_ordered(self, es_doc):
            keywords = self.parser.get_keywords_tf_idf_denormalized(self.index, es_doc["_id"], es_doc["_source"]["title"], es_doc["_source"]["text"], keep_order=True)
            query = " ".join(keywords)
            return self.em.encode(query)

    def get_features(self, query_es, doc_id, bm25_score=None, cosine_score=None):
        doc_es = self.es.get(index=self.index, id=doc_id)

        doc_length = len(doc_es["_source"]["title"]) + len(doc_es["_source"]["text"])
        query_published_after = 1 if int(query_es["_source"]["date"]) > int(doc_es["_source"]["date"]) else 0

        if not bm25_score:
            bm25_score = 0
            query_keywords = self.parser.get_keywords_tf_idf(self.index, query_es["_id"])
            query = " OR ".join(query_keywords)
            if query:
                val = self.es.explain(
                    index=self.index,
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
            query_emb = self.get_embedding_of_extracted_keywords_denormalized_ordered(query_es)
            doc_emb = self.get_embedding_of_extracted_keywords_denormalized_ordered(doc_es)
            if query_emb and doc_emb:
                cosine_score = cosine(query_emb, doc_emb)
        
        return [bm25_score, cosine_score, doc_length, query_published_after]

    def get_training_and_validation_set(self, data):
        np.random.shuffle(data)
        train_size = int(len(data) * 0.7)
        train_data_raw = data[:train_size]
        val_data_raw = data[train_size:]
        return (train_data_raw, val_data_raw)

    def split_training_data(self, data):
        X = []
        y = []
        query_groups = []
        for jl in tqdm(data, total=len(data)):
            query_es = self.es.get(index=self.index, id=jl["id"])
            query_groups.append(len(jl["references"]))
            for ref in jl["references"]:
                ref_features = self.get_features(query_es, ref["id"])
                X.append(ref_features)
                y.append(int(ref["exp_rel"]))
        return (X,y,query_groups)

    def get_training_data(self):
        data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
        judgement_list_paths = [f"{data_location}/judgement_list_wapo_18.jsonl", f"{data_location}/judgement_list_wapo_19.jsonl"]
        data = []
        for path in judgement_list_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    judgement = json.loads(line)
                    data.append(judgement)
        train_data_raw, val_data_raw = self.get_training_and_validation_set(data)
        X_train, y_train, query_train = self.split_training_data(train_data_raw)
        X_val, y_val, query_val = self.split_training_data(val_data_raw)
        return (np.array(X_train), np.array(y_train), np.array(query_train), np.array(X_val), np.array(y_val), np.array(query_val))

    def get_combined_retrieval(self, size, query_es):
        results = []
        keywords = self.parser.get_keywords_tf_idf(self.index, query_es["_id"])
        if keywords:
            query_keywords = " OR ".join(keywords)
            k_results = (self.es.search(
                size = size,
                index = self.index,
                body = {
                    "query": {
                        "query_string": {
                            "fields": [ "title", "text" ],
                            "query": query_keywords
                        }
                    }
                }
            ))["hits"]["hits"]
            results = [{"id": res["_id"], "bm25_score":res["_score"], "cosine_score":None} for res in k_results]
        keywords_denorm = self.parser.get_keywords_tf_idf_denormalized(self.index, query_es["_id"], query_es["_source"]["title"], query_es["_source"]["text"], keep_order=True)
        if keywords_denorm:
            emb_query = self.em.encode(" ".join(keywords_denorm))
            nearest_n: NearestNeighborList = self.vs.get_k_nearest(emb_query,size)
            e_results = [{"id": list(nn.keys())[0], "cosine_score":list(nn.values())[0], "bm25_score":None} for nn in nearest_n[0]]
            for res_e in e_results:
                is_new_res = True
                for i, res_k in enumerate(results):
                    if res_e["id"] == res_k["id"]:
                        results[i]["cosine_score"] = res_e["cosine_score"]
                        is_new_res = False
                        break
                if is_new_res:
                    results.append(res_e)
        return results

    def test_model(self, size, judgement_list_path, result_path, model):
        topic_dict = {v: k for k, v in (JudgementListWapo.get_topic_dict("20")).items()}
        print("Retrieve background links for each topic and calculate features...")
        with open(judgement_list_path, "r", encoding="utf-8") as f:
            with open(result_path, "w", encoding="utf-8") as fout:
                for line in tqdm(f, total=50):
                    judgement = json.loads(line)
                    query_es = self.es.get(index=self.index,id=judgement["id"])
                    combined_retrieval = self.get_combined_retrieval(size, query_es)
                    X_test = []
                    X_test_ids = []
                    for res in combined_retrieval:
                        res_features = self.get_features(query_es,res["id"],res["bm25_score"],res["cosine_score"])
                        X_test.append(res_features)
                        X_test_ids.append(res["id"])
                    X_test = np.array(X_test)
                    X_test_ids = np.array(X_test_ids)
                    test_pred = model.predict(X_test)
                    inds = (test_pred.argsort())[::-1]
                    ranked_test_pred = test_pred[inds]
                    ranked_retrieval = X_test_ids[inds]
                    topic=topic_dict[judgement["id"]]
                    for rank, ret in enumerate(ranked_retrieval):
                        out = f"{topic}\tQ0\t{ret}\t{rank}\t{ranked_test_pred[rank]}\tducrun\n"
                        fout.write(out)
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
    ranker = WAPORanker(es, parser, em, vs, index)

    if not os.path.isfile(f"{data_location}/ranking_model.txt"):
        if not os.path.isfile(f"{data_location}/X_train.txt"):
            print("Initialize training and validation data...")
            X_train, y_train, query_train, X_val, y_val, query_val = ranker.get_training_data()

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

    model_path = f"{data_location}/ranking_model.txt"
    if os.path.isfile(model_path):
        print("Load model from file...")
        model = lgb.Booster(model_file=model_path)
        result_path = f"{data_location}/ranking_results_20.txt"
        ranker.test_model(300, judgement_list_20_path, result_path, model)