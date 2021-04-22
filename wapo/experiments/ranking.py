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
        self.data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"

    def get_embedding_of_extracted_keywords_denormalized_ordered(self, es_doc):
            keywords = self.parser.get_keywords_tf_idf_denormalized(self.index, es_doc["_id"], es_doc["_source"]["title"], es_doc["_source"]["text"], keep_order=True)
            query = " ".join(keywords)
            return self.em.encode(query)

    def get_features(self, query_es, doc_es, bm25_score=None, cosine_score=None):
        doc_length = len(doc_es["_source"]["title"]) + len(doc_es["_source"]["text"])
        query_published_after = 1 if int(query_es["_source"]["date"]) > int(doc_es["_source"]["date"]) else 0

        if not bm25_score:
            bm25_score = 0
            query_keywords = self.parser.get_keywords_tf_idf(self.index, query_es["_id"])
            query = " OR ".join(query_keywords)
            if query:
                val = self.es.explain(
                    index=self.index,
                    id=doc_es["_id"],
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
            cosine_score = 0
            query_emb = self.get_embedding_of_extracted_keywords_denormalized_ordered(query_es)
            doc_emb = self.get_embedding_of_extracted_keywords_denormalized_ordered(doc_es)
            if query_emb and doc_emb:
                cosine_score = 1 - cosine(query_emb, doc_emb) # convert cosine sim. to cosine dist. as trev_eval sorts in desc. order
        
        return np.array([bm25_score, cosine_score, doc_length, query_published_after])

    def get_training_and_validation_set(self, data):
        np.random.seed(69)
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
            count = 0
            for ref in jl["references"]:
                if ref["id"] == jl["id"]:
                    continue
                doc_es = None
                try:
                    doc_es = self.es.get(
                        index = self.index,
                        id = ref["id"]
                    )
                except Exception as e:
                    print(e)
                    continue
                ref_features = self.get_features(query_es, doc_es)
                X.append(ref_features)
                y.append(int(ref["exp_rel"]))
                count += 1
            query_groups.append(count)
        return (X,y,query_groups)

    def get_training_data(self, jl_paths):
        data = []
        for path in jl_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    judgement = json.loads(line)
                    data.append(judgement)
        train_data_raw, val_data_raw = self.get_training_and_validation_set(data)
        X_train, y_train, query_train = self.split_training_data(train_data_raw)
        X_val, y_val, query_val = self.split_training_data(val_data_raw)
        return (np.array(X_train), np.array(y_train), np.array(query_train), np.array(X_val), np.array(y_val), np.array(query_val))

    def get_combined_retrieval(self, size, query_es):
        results = self.get_ranked_bool_retrieval(size, query_es)
        results_sem_search = self.get_semantic_search_retrieval(size, query_es)
        orig_len = len(results)
        for res in results_sem_search:
            is_new_res = True
            for i in range(orig_len):
                if res["id"] == results[i]["id"]:
                    results[i]["cosine_score"] = res["cosine_score"]
                    is_new_res = False
                    break
            if is_new_res:
                results.append(res)
        return results

    def get_ranked_bool_retrieval(self, size, query_es):
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
            results = [{"id": res["_id"], "bm25_score":res["_score"], "cosine_score":None} for res in k_results if res["_id"] != query_es["_id"]]
        return results

    def get_semantic_search_retrieval(self, size, query_es):
        results = []
        keywords_denorm = self.parser.get_keywords_tf_idf_denormalized(self.index, query_es["_id"], query_es["_source"]["title"], query_es["_source"]["text"], keep_order=True)
        if keywords_denorm:
            emb_query = self.em.encode(" ".join(keywords_denorm))
            nearest_n: NearestNeighborList = self.vs.get_k_nearest(emb_query,size)
            results = [{"id": list(nn.keys())[0], "cosine_score":1-list(nn.values())[0], "bm25_score":None} for nn in nearest_n[0] if list(nn.keys())[0] != query_es["_id"]] # convert cosine sim. to cosine dist. as trev_eval sorts in desc. order
        return results

    def get_ranking(self, test_pred, test_ids):
        inds = (test_pred.argsort())[::-1]
        ranked_test_pred = test_pred[inds]
        ranked_ids = test_ids[inds]
        return (ranked_test_pred, ranked_ids)

    def test_model(self, retrieval_func, feature_inds, jl_path, res_path, model):
        topic_dict = {v: k for k, v in (JudgementListWapo.get_topic_dict("20")).items()}
        print("Retrieve background links for each topic and calculate features...")
        with open(jl_path, "r", encoding="utf-8") as f:
            with open(res_path, "w", encoding="utf-8") as fout:
                for line in tqdm(f, total=49):
                    judgement = json.loads(line)
                    query_es = self.es.get(index=self.index,id=judgement["id"])
                    X_test = []
                    X_test_ids = []
                    retrieval = retrieval_func(query_es)
                    for res in retrieval:
                        doc_es = None
                        try:
                            doc_es = self.es.get(
                                index = self.index,
                                id = res["id"]
                            )
                        except Exception as e:
                            print(e)
                            continue
                        res_features = self.get_features(query_es,doc_es,res["bm25_score"],res["cosine_score"])[feature_inds]
                        X_test.append(res_features)
                        X_test_ids.append(res["id"])
                    X_test = np.array(X_test)
                    X_test_ids = np.array(X_test_ids)
                    test_pred = model.predict(X_test)
                    ranked_test_pred, ranked_ids = self.get_ranking(test_pred, X_test_ids)
                    topic = topic_dict[judgement["id"]]
                    for rank, ret in enumerate(ranked_ids):
                        out = f"{topic}\tQ0\t{ret}\t{rank}\t{ranked_test_pred[rank]}\tducrun\n"
                        fout.write(out)

    def test_baseline(self, size, jl_path, result_path):
        topic_dict = {v: k for k, v in (JudgementListWapo.get_topic_dict("20")).items()}
        print("Retrieve background links for each topic and rank based on ranked Boolean...")
        with open(jl_path, "r", encoding="utf-8") as f:
            with open(result_path, "w", encoding="utf-8") as fout:
                for line in tqdm(f, total=49):
                    judgement = json.loads(line)
                    query_es = self.es.get(index=self.index,id=judgement["id"])
                    retrieval = self.get_ranked_bool_retrieval(size, query_es) # already ranked
                    bm25_scores = np.array([ret["bm25_score"] for ret in retrieval])
                    ret_ids = np.array([ret["id"] for ret in retrieval])
                    topic = topic_dict[judgement["id"]]
                    for rank, rid in enumerate(ret_ids):
                        out = f"{topic}\tQ0\t{rid}\t{rank}\t{bm25_scores[rank]}\tducrun\n"
                        fout.write(out)

    def test_semantic_search(self, size, jl_path, result_path):
        topic_dict = {v: k for k, v in (JudgementListWapo.get_topic_dict("20")).items()}
        print("Retrieve background links for each topic and rank based on semantic search cosine similarity...")
        with open(jl_path, "r", encoding="utf-8") as f:
            with open(result_path, "w", encoding="utf-8") as fout:
                for line in tqdm(f, total=49):
                    judgement = json.loads(line)
                    query_es = self.es.get(index=self.index,id=judgement["id"])
                    retrieval = self.get_semantic_search_retrieval(size, query_es) # already ranked
                    cosine_scores = np.array([ret["cosine_score"] for ret in retrieval])
                    ret_ids = np.array([ret["id"] for ret in retrieval])
                    topic = topic_dict[judgement["id"]]
                    for rank, rid in enumerate(ret_ids):
                        out = f"{topic}\tQ0\t{rid}\t{rank}\t{cosine_scores[rank]}\tducrun\n"
                        fout.write(out)

    def get_combined_reversed_topic_dict(self):
        topic_dict_18 = JudgementListWapo.get_topic_dict('18')
        topic_dict_19 = JudgementListWapo.get_topic_dict('19')
        topic_dict_20 = JudgementListWapo.get_topic_dict('20')
        topic_dict_combined = {**topic_dict_18, **topic_dict_19, **topic_dict_20}
        topic_dict_combined = {v:k for k,v in topic_dict_combined.items()}
        return topic_dict_combined

    def rank_by_features_individually(self, judgement_list_path, result_file_name):
        topic_dict_combined = self.get_combined_reversed_topic_dict()
        exception_count = 0
        with open(judgement_list_path, "r", encoding="utf-8") as f:
            jl = []
            for line in tqdm(f, total = 107):
                judgm = json.loads(line)
                j = {"id": judgm["id"], "references": []}
                try:
                    query_es = es.get(index=self.index, id=judgm["id"])
                    for ref in judgm["references"]:
                        if ref["id"] == judgm["id"]:
                            continue
                        doc_es = None
                        try:
                            doc_es = self.es.get(
                                index = self.index,
                                id = ref["id"]
                            )
                        except Exception as e:
                            print(e)
                            continue
                        feat = self.get_features(query_es, doc_es)
                        j["references"].append({"id": ref["id"], "features": feat})
                    jl.append(j)
                except:
                    exception_count += 1
                    continue
            with open(f"{self.data_location}/{result_file_name}_bm25.txt", "w", encoding="utf-8") as fout_bm25:
                with open(f"{self.data_location}/{result_file_name}_cos.txt", "w", encoding="utf-8") as fout_cos:
                    for j in jl:
                        bm25_scores = np.array([ref["features"][0] for ref in j["references"]])
                        cos_scores = np.array([ref["features"][1] for ref in j["references"]])
                        ref_ids = np.array([ref["id"] for ref in j["references"]])
                        ranked_bm25_scores, ranked_bm25_refs = self.get_ranking(bm25_scores,ref_ids)
                        ranked_cos_scores, ranked_cos_refs = self.get_ranking(cos_scores,ref_ids)
                        topic = topic_dict_combined[j["id"]]
                        for rank, ret in enumerate(ranked_bm25_refs):
                            out = f"{topic}\tQ0\t{ret}\t{rank}\t{ranked_bm25_scores[rank]}\tducrun\n"
                            fout_bm25.write(out)
                        for rank, ret in enumerate(ranked_cos_refs):
                            out = f"{topic}\tQ0\t{ret}\t{rank}\t{ranked_cos_scores[rank]}\tducrun\n"
                            fout_cos.write(out)
            print(f"Exception count: {exception_count}")

    def experiment_assume_perfect_recall(self, jl_paths, res_names):
        print("Start ranking experiment assuming perfect recall...")
        for i, path in enumerate(jl_paths):
            self.rank_by_features_individually(path, res_names[i])

    def experiment_baseline_ranking(self, jl_path):
        print("Test baseline ranking...")
        ret_count = [100,150,200,250,300]
        for ret in ret_count:
            result_path = f"{self.data_location}/wapo_ranking_base_{str(ret)}.txt"
            self.test_baseline(ret,jl_path, result_path)

    def experiment_semantic_search_ranking(self, jl_path):
        print("Test semantic search ranking...")
        ret_count = [100,150,200,250,300]
        for ret in ret_count:
            result_path = f"{self.data_location}/wapo_ranking_cos_{str(ret)}.txt"
            self.test_semantic_search(ret,jl_path, result_path)

    def experiment_rank_feature_pairs(self, X_train, y_train, query_train, X_val, y_val, query_val, jl_path):
        ret_count = [100,150,200]
        for ret in ret_count:
            print(f"Test baseline + cosine similarity model. k = {ret}")
            def get_combined_retrieval(query_es):
                return self.get_combined_retrieval(ret//2, query_es)

            model_combined = lgb.LGBMRanker()
            model_combined.fit(
                np.array([v[:2] for v in X_train]),
                y_train,
                group=query_train,
                eval_set=[(np.array([v[:2] for v in X_val]), y_val)],
                eval_group=[query_val],
                eval_at=[5,10],
                early_stopping_rounds=50
            )
            self.test_model(get_combined_retrieval, [0,1], jl_path, f"{self.data_location}/wapo_ranking_base_cos_{ret}.txt", model_combined)

            print(f"Test baseline + doc length model. k = {ret}")
            def get_retrieval_base_doc_len(query_es):
                return self.get_ranked_bool_retrieval(ret, query_es)

            model_base_doc_len = lgb.LGBMRanker()
            model_base_doc_len.fit(
                np.array([[v[0], v[2]] for v in X_train]),
                y_train,
                group=query_train,
                eval_set=[(np.array([[v[0], v[2]] for v in X_val]), y_val)],
                eval_group=[query_val],
                eval_at=[5,10],
                early_stopping_rounds=50
            )
            self.test_model(get_retrieval_base_doc_len, [0,2], jl_path, f"{data_location}/wapo_ranking_base_doc_len_{ret}.txt", model_base_doc_len)

            print(f"Test baseline + time model. k = {ret}")
            def get_retrieval_base_time(query_es):
                return self.get_ranked_bool_retrieval(ret, query_es)

            model_base_time = lgb.LGBMRanker()
            model_base_time.fit(
                np.array([[v[0], v[3]] for v in X_train]),
                y_train,
                group=query_train,
                eval_set=[(np.array([[v[0], v[3]] for v in X_val]), y_val)],
                eval_group=[query_val],
                eval_at=[5,10],
                early_stopping_rounds=50
            )
            self.test_model(get_retrieval_base_time, [0,3], jl_path, f"{data_location}/wapo_ranking_base_time_{ret}.txt", model_base_time)

            print(f"All features combined. k = {ret}")
            model_all = lgb.LGBMRanker()
            model_all.fit(
                X_train,
                y_train,
                group=query_train,
                eval_set=[(X_val, y_val)],
                eval_group=[query_val],
                eval_at=[5,10],
                early_stopping_rounds=50
            )
            self.test_model(get_combined_retrieval, [0,1,2,3], jl_path, f"{self.data_location}/wapo_ranking_all_{ret}.txt", model_all)

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
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    judgement_list_18_path = f"{data_location}/judgement_list_wapo_18.jsonl"
    judgement_list_19_path = f"{data_location}/judgement_list_wapo_19.jsonl"
    judgement_list_20_path = f"{data_location}/judgement_list_wapo_20.jsonl"
    vs_extracted_k_denormalized_ordered = f"{data_location}/wapo_vs_extracted_k_denormalized_ordered.bin"
    vs = VectorStorage(vs_extracted_k_denormalized_ordered)
    ranker = WAPORanker(es, parser, em, vs, index)

    if not os.path.isfile(f"{data_location}/X_train.txt"):
        print("Initialize training and validation data...")
        X_train, y_train, query_train, X_val, y_val, query_val = ranker.get_training_data([judgement_list_18_path, judgement_list_19_path])

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

    ranker.experiment_assume_perfect_recall(
        [judgement_list_18_path, judgement_list_19_path, judgement_list_20_path],
        ["wapo_ranking_perf_recall_18", "wapo_ranking_perf_recall_19", "wapo_ranking_perf_recall_20"]
    )
    ranker.experiment_baseline_ranking(judgement_list_20_path) # output file: wapo_ranking_base_{k}.txt
    ranker.experiment_semantic_search_ranking(judgement_list_20_path) # output file: wapo_ranking_cos_{k}.txt
    ranker.experiment_rank_feature_pairs(X_train, y_train, query_train, X_val, y_val, query_val, judgement_list_20_path) # output files: wapo_ranking_base_*_{k}.txt