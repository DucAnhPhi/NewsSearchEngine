import argparse
import json
import os
from elasticsearch import Elasticsearch
from ..parser import ParserNetzpolitik
from ...vector_storage import VectorStorage
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...typings import NearestNeighborList

class CombinedRecallExperiment():
    def __init__(self, es, parser, index, size, get_keywords_query_func, get_embedding_query_func, vector_storage_location, judgement_list_path):
        self.es = es
        self.parser = parser
        self.index = index
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.
        self.min_recall = 1.
        self.max_recall = 0.
        self.exception_count = 0
        self.add_count_avg = 0
        self.recall_improvement_avg = 0.

        # load vector storage from file
        self.vs = VectorStorage(vector_storage_location, 20000)

        with open(judgement_list_path, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                try:
                    result_ids = []
                    query_article_es = self.es.get(
                        index = self.index,
                        id = judgement["id"]
                    )
                    self.count += 1
                    keywords_query = get_keywords_query_func(query_article_es)
                    if keywords_query:
                        k_results = (self.es.search(
                            size = size,
                            index = self.index,
                            body = {
                                "query": {
                                "multi_match": {
                                    "fields": [ "title", "body" ],
                                    "query": keywords_query,
                                    "analyzer": "german",
                                    "operator": "or"
                                    }
                                }
                            }
                        ))["hits"]["hits"]
                        result_ids = [res["_id"] for res in k_results]
                    embedding_query = get_embedding_query_func(query_article_es)
                    if embedding_query:
                        nearest_n: NearestNeighborList = self.vs.get_k_nearest(embedding_query,size)
                        e_results = [list(nn.keys())[0] for nn in nearest_n[0]]
                        for res in e_results:
                            if res not in result_ids:
                                for ref in judgement["references"]:
                                    if ref == res:
                                        self.add_count_avg += 1
                                        self.recall_improvement_avg += 1/len(judgement["references"])
                                        break
                                result_ids.append(res)
                    recall = 0.
                    self.retrieval_count_avg += len(result_ids)
                    for res_id in result_ids:
                        if res_id == judgement["id"]:
                            continue
                        if res_id in judgement["references"]:
                            recall += 1
                    recall /= len(judgement["references"])
                    self.recall_avg += recall
                    if recall < self.min_recall:
                        self.min_recall = recall
                    if recall > self.max_recall:
                        self.max_recall = recall
                except Exception as e:
                    # query article not found
                    self.exception_count += 1
                    print(e)
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count
        self.add_count_avg /= self.count
        self.recall_improvement_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")
        print(f"Min Recall: {self.min_recall}")
        print(f"Max Recall: {self.max_recall}")
        print(f"Exception Count: {self.exception_count}")
        print(f"Avg Recall improvement with semantic search: {self.recall_improvement_avg}")
        print(f"Avg different results introduced with semantic search: {self.add_count_avg}")

if __name__ == "__main__":
    index = "netzpolitik"

    p = argparse.ArgumentParser(description='Run netzpolitik.org combined retrieval experiments')
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

    parser = ParserNetzpolitik(es)
    em = EmbeddingModel(lang="de", device=args.device)
    fe = FeatureExtraction(em, parser)
    size = 300
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    judgement_list_path = f"{data_location}/judgement_list_netzpolitik.jsonl"
    vs_title_with_first_paragraph = f"{data_location}/netzpolitik_vs_title_with_first_paragraph.bin"
    vs_annotated_k = f"{data_location}/netzpolitik_vs_annotated_k.bin"

    def get_embedding_of_title_with_first_paragraph(es_doc):
        return fe.get_embedding_of_title_with_first_paragraph(es_doc["_source"])

    def get_embedding_of_annotated_keywords(es_doc):
        keywords = es_doc["_source"]["keywords"]
        query = " ".join(keywords)
        if not query:
            return None
        return em.encode(query)

    def get_query_from_annotated_and_tf_idf_keywords(es_doc):
        annotated = es_doc["_source"]["keywords"]
        extracted = parser.get_keywords_tf_idf(args.index_name, es_doc["_id"])
        return " ".join(annotated + extracted)

    exp = CombinedRecallExperiment(
        es,
        parser,
        index,
        size,
        get_query_from_annotated_and_tf_idf_keywords,
        get_embedding_of_annotated_keywords,
        vs_annotated_k,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Run combined retrieval method using keyword matching and semantic search.")
    print("Semantic Search configuration:")
    print("Index articles by:   embedding of annotated keywords")
    print("Query:               embedding of annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    exp2 = CombinedRecallExperiment(
        es,
        parser,
        index,
        size,
        get_query_from_annotated_and_tf_idf_keywords,
        get_embedding_of_title_with_first_paragraph,
        vs_title_with_first_paragraph,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Run combined retrieval method using keyword matching and semantic search.")
    print("Semantic Search configuration:")
    print("Index articles by:   embedding of title with first paragraph")
    print("Query:               embedding of title with first paragraph")
    exp2.print_stats()
    print("----------------------------------------------------------------\n")
