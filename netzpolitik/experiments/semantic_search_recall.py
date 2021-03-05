import os.path
import pathlib
import json
import argparse
from elasticsearch import Elasticsearch
from ...vector_storage import VectorStorage
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...typings import NearestNeighborList
from ..parser import ParserNetzpolitik

class SemanticSearchExperiment():
    def __init__(self, es, index, size, get_query_func, vector_storage_location, judgement_list_path):
        self.es = es
        self.index = index
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.
        self.min_recall = 1.
        self.max_recall = 0.

        # load vector storage from file
        self.vs = VectorStorage(vector_storage_location, 500000)

        with open(judgement_list_path, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                try:
                    query_article_es = (self.es.search(
                        size = 1,
                        index = self.index,
                        body = {
                            "query": {
                                "term": {
                                    "url": {
                                        "value": judgement["id"]
                                    }
                                }
                            }
                        }
                    ))["hits"]["hits"][0]
                    query = get_query_func(query_article_es)
                    if not query:
                        continue
                    self.count += 1
                    nearest_n: NearestNeighborList = self.vs.get_k_nearest(query,size)
                    result_ids = [list(nn.keys())[0] for nn in nearest_n[0]]
                    recall = 0.
                    self.retrieval_count_avg += len(result_ids)
                    for res_id in result_ids:
                        res_url = self.es.get(
                            index = self.index,
                            id=res_id
                        )["_source"]["url"]
                        if res_url == judgement["id"]:
                            continue
                        if res_url in judgement["references"]:
                            recall += 1
                    recall /= len(judgement["references"])
                    self.recall_avg += recall
                    if recall < self.min_recall:
                        self.min_recall = recall
                    if recall > self.max_recall:
                        self.max_recall = recall
                except:
                    # query article not found
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")
        print(f"Min Recall: {self.min_recall}")
        print(f"Max Recall: {self.max_recall}")

if __name__ == "__main__":
    index = "netzpolitik"
    p = argparse.ArgumentParser(description='Run netzpolitik.org semantic search recall experiments')
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
    size = 100
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    judgement_list_path = f"{data_location}/judgement_list_netzpolitik.jsonl"
    vs_title = f"{data_location}/netzpolitik_vs_title.bin"
    vs_title_with_section_titles = f"{data_location}/netzpolitik_vs_title_with_section_titles.bin"
    vs_title_with_first_paragraph = f"{data_location}/netzpolitik_vs_title_with_first_paragraph.bin"
    vs_annotated_k = f"{data_location}/netzpolitik_vs_annotated_k.bin"
    vs_extracted_k_normalized = f"{data_location}/netzpolitik_vs_extracted_k_normalized.bin"
    vs_extracted_k_denormalized = f"{data_location}/netzpolitik_vs_extracted_k_denormalized.bin"
    vs_extracted_k_denormalized_ordered = f"{data_location}/netzpolitik_vs_extracted_k_denormalized_ordered.bin"

    def get_embedding_of_title(es_doc):
        return fe.get_embedding_of_title(es_doc["_source"])

    def get_embedding_of_title_with_first_paragraph(es_doc):
        return fe.get_embedding_of_title_with_first_paragraph(es_doc["_source"])

    def get_embedding_of_title_with_section_titles(es_doc):
        return fe.get_embedding_of_title_with_section_titles(es_doc["_source"])

    def get_embedding_of_annotated_keywords(es_doc):
        keywords = es_doc["_source"]["keywords"]
        query = " ".join(keywords)
        if not query:
            return None
        return em.encode(query)

    # Index articles by:    embedding of title
    # Query:                embedding of title
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title,
        vs_title,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title")
    print("Query:               embedding of title")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title
    # Query:                embedding of title w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title_with_first_paragraph,
        vs_title,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title")
    print("Query:               embedding of title w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title
    # Query:                embedding of title w/ section titles
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title_with_section_titles,
        vs_title,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title")
    print("Query:               embedding of title w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title
    # Query:                embedding of pre-annotated keywords
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_annotated_keywords,
        vs_title,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ first paragraph
    # Query:                embedding of title w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title_with_first_paragraph,
        vs_title_with_first_paragraph,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ first paragraph")
    print("Query:               embedding of title w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ first paragraph
    # Query:                embedding of title
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title,
        vs_title_with_first_paragraph,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ first paragraph")
    print("Query:               embedding of title")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ first paragraph
    # Query:                embedding of title w/ section titles
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title_with_section_titles,
        vs_title_with_first_paragraph,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ first paragraph")
    print("Query:               embedding of title w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ first paragraph
    # Query:                embedding of pre-annotated keywords
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_annotated_keywords,
        vs_title_with_first_paragraph,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ first paragraph")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ section titles
    # Query:                embedding of title w/ section titles
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title_with_section_titles,
        vs_title_with_section_titles,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ section titles")
    print("Query:               embedding of title w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ section titles
    # Query:                embedding of title
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title,
        vs_title_with_section_titles,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ section titles")
    print("Query:               embedding of title")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ section titles
    # Query:                embedding of title w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title_with_first_paragraph,
        vs_title_with_section_titles,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ section titles")
    print("Query:               embedding of title w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ section titles
    # Query:                embedding of pre-annotated keywords
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_annotated_keywords,
        vs_title_with_section_titles,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ section titles")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of pre-annotated keywords
    # Query:                embedding of pre-annotated keywords
    # Recall Avg:           0.110762
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_annotated_keywords,
        vs_annotated_k,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of pre-annotated keywords")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of pre-annotated keywords
    # Query:                embedding of title
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title,
        vs_annotated_k,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of pre-annotated keywords")
    print("Query:               embedding of title")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of pre-annotated keywords
    # Query:                embedding of title w/ section titles
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title_with_section_titles,
        vs_annotated_k,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of pre-annotated keywords")
    print("Query:               embedding of title w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of pre-annotated keywords
    # Query:                embedding of title w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_title_with_first_paragraph,
        vs_annotated_k,
        judgement_list_path
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of pre-annotated keywords")
    print("Query:               embedding of title w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")