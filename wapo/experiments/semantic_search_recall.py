import argparse
import json
import os
from elasticsearch import Elasticsearch
from ..parser import ParserWAPO
from ...vector_storage import VectorStorage
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...typings import NearestNeighborList

class SemanticSearchExperiment():
    def __init__(self, es, index, size, get_query_func, vector_storage_location, judgement_list_path, rel_cutoff):
        self.es = es
        self.index = index
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.
        self.min_recall = 1.
        self.max_recall = 0.
        self.rel_cutoff = rel_cutoff
        self.exception_count = 0

        # load vector storage from file
        self.vs = VectorStorage(vector_storage_location, 500000)

        with open(judgement_list_path, "r", encoding="utf-8") as f:
            for line in f:
                judgement = json.loads(line)
                # apply relevance cutoff
                relevant_articles = [ref for ref in judgement["references"] if int(ref["exp_rel"]) >= self.rel_cutoff]
                if len(relevant_articles) == 0:
                    self.count += 1
                    self.min_recall = 0
                    continue
                try:
                    query_article_es = self.es.get(
                        index = self.index,
                        id = judgement["id"]
                    )
                    query = get_query_func(query_article_es)
                    if not query:
                        continue
                    self.count += 1
                    nearest_n: NearestNeighborList = self.vs.get_k_nearest(query,size)
                    result_ids = [list(nn.keys())[0] for nn in nearest_n[0]]
                    recall = 0.
                    self.retrieval_count_avg += len(result_ids)
                    for res_id in result_ids:
                        if res_id == judgement["id"]:
                            continue
                        for ref in relevant_articles:
                            if ref["id"] == res_id:
                                recall += 1
                                break
                    recall /= len(relevant_articles)
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

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")
        print(f"Min Recall: {self.min_recall}")
        print(f"Max Recall: {self.max_recall}")
        print(f"Exception Count: {self.exception_count}")

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
    size = 100
    rel_cutoff = 4
    data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
    judgement_list_path = f"{data_location}/judgement_list_wapo.jsonl"
    vs_title = f"{data_location}/wapo_vs_title.bin"
    vs_title_with_section_titles = f"{data_location}/wapo_vs_title_with_section_titles.bin"
    vs_title_with_first_paragraph = f"{data_location}/wapo_vs_title_with_first_paragraph.bin"
    vs_extracted_k_normalized = f"{data_location}/wapo_vs_extracted_k_normalized.bin"
    vs_extracted_k_denormalized = f"{data_location}/wapo_vs_extracted_k_denormalized.bin"
    vs_extracted_k_denormalized_ordered = f"{data_location}/wapo_vs_extracted_k_denormalized_ordered.bin"

    def get_embedding_of_title(es_doc):
        return fe.get_embedding_of_title(es_doc["_source"])

    def get_embedding_of_title_with_first_paragraph(es_doc):
        return fe.get_embedding_of_title_with_first_paragraph(es_doc["_source"])

    def get_embedding_of_title_with_section_titles(es_doc):
        return fe.get_embedding_of_title_with_section_titles(es_doc["_source"])

    def get_embedding_of_extracted_keywords_normalized(es_doc):
        keywords = parser.get_keywords_tf_idf(index, es_doc["_id"])
        query = " ".join(keywords)
        if not query:
            return None
        return em.encode(query)

    def get_embedding_of_extracted_keywords_denormalized(es_doc):
        keywords = parser.get_keywords_tf_idf_denormalized(index, es_doc["_id"], es_doc["_source"]["text"], keep_order=False)
        query = " ".join(keywords)
        if not query:
            return None
        return em.encode(query)

    def get_embedding_of_extracted_keywords_denormalized_ordered(es_doc):
        keywords = parser.get_keywords_tf_idf_denormalized(index, es_doc["_id"], es_doc["_source"]["text"], keep_order=True)
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
        judgement_list_path,
        rel_cutoff
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
        judgement_list_path,
        rel_cutoff
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
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title")
    print("Query:               embedding of title w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title
    # Query:                embedding of extracted keywords (normalized)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_normalized,
        vs_title,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title")
    print("Query:               embedding of extracted keywords (normalized)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title
    # Query:                embedding of extracted keywords (denormalized)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_denormalized,
        vs_title,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title")
    print("Query:               embedding of extracted keywords (denormalized)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title
    # Query:                embedding of extracted keywords (denormalized, order preserved)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_denormalized_ordered,
        vs_title,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title")
    print("Query:               embedding of extracted keywords (denormalized, order preserved)")
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
        judgement_list_path,
        rel_cutoff
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
        judgement_list_path,
        rel_cutoff
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
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ first paragraph")
    print("Query:               embedding of title w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ first paragraph
    # Query:                embedding of extracted keywords (normalized)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_normalized,
        vs_title_with_first_paragraph,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ first paragraph")
    print("Query:               embedding of extracted keywords (normalized)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ first paragraph
    # Query:                embedding of extracted keywords (denormalized)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_denormalized,
        vs_title_with_first_paragraph,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ first paragraph")
    print("Query:               embedding of extracted keywords (denormalized)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ first paragraph
    # Query:                embedding of extracted keywords (denormalized, order preserved)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_denormalized_ordered,
        vs_title_with_first_paragraph,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ first paragraph")
    print("Query:               embedding of extracted keywords (denormalized, order preserved)")
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
        judgement_list_path,
        rel_cutoff
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
        judgement_list_path,
        rel_cutoff
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
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ section titles")
    print("Query:               embedding of title w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ section titles
    # Query:                embedding of extracted keywords (normalized)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_normalized,
        vs_title_with_section_titles,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ section titles")
    print("Query:               embedding of extracted keywords (normalized)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ section titles
    # Query:                embedding of extracted keywords (denormalized)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_denormalized,
        vs_title_with_section_titles,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ section titles")
    print("Query:               embedding of extracted keywords (denormalized)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of title w/ section titles
    # Query:                embedding of extracted keywords (denormalized, order preserved)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_denormalized_ordered,
        vs_title_with_section_titles,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of title w/ section titles")
    print("Query:               embedding of extracted keywords (denormalized, order preserved)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of extracted keywords (normalized)
    # Query:                embedding of extracted keywords (normalized)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_normalized,
        vs_extracted_k_normalized,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of extracted keywords (normalized)")
    print("Query:               embedding of extracted keywords (normalized)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of extracted keywords (denormalized)
    # Query:                embedding of extracted keywords (denormalized)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_denormalized,
        vs_extracted_k_denormalized,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of extracted keywords (denormalized)")
    print("Query:               embedding of extracted keywords (denormalized)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of extracted keywords (denormalized, order preserved)
    # Query:                embedding of extracted keywords (denormalized, order preserved)
    exp = SemanticSearchExperiment(
        es,
        args.index_name,
        size,
        get_embedding_of_extracted_keywords_denormalized_ordered,
        vs_extracted_k_denormalized_ordered,
        judgement_list_path,
        rel_cutoff
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of extracted keywords (denormalized, order preserved)")
    print("Query:               embedding of extracted keywords (denormalized, order preserved)")
    exp.print_stats()
    print("----------------------------------------------------------------\n")