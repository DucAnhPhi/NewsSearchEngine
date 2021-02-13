import os.path
import pathlib
import json
from elasticsearch import Elasticsearch
from ...vector_storage import VectorStorage
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...typings import NearestNeighborList
from ..parser import ParserNetzpolitik

class SemanticSearchExperiment():
    def __init__(self, es, parser, query_emb_func, storage_file, keywords_tf_idf=False, size=100):
        self.es = es
        self.parser = parser
        self.index = "netzpolitik"
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.

        # init vector storage
        data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
        storage_location = f"{data_location}/{storage_file}"
        print("Loading vector storage from file...\n")
        self.vs = VectorStorage(storage_location)

        # build query
        with open(f"{data_location}/judgement_list_netzpolitik.jsonl", "r", encoding="utf-8") as f:
            for line in (f):
                judgment = json.loads(line)
                try:
                    self.count += 1
                    query_article = (self.es.get(
                        index=self.index,
                        id=judgment["id"]
                    ))["_source"]
                    if keywords_tf_idf:
                        query_article["keywords"] = self.parser.get_keywords_tf_idf_de(self.index, judgment["id"])
                    query = query_emb_func(query_article)
                    if query == None:
                        self.count -= 1
                        continue
                    nearest_n: NearestNeighborList = self.vs.get_k_nearest(query,size)
                    result_ids = [list(nn.keys())[0] for nn in nearest_n[0]]
                    recall = 0.
                    self.retrieval_count_avg += len(result_ids)
                    for res_id in result_ids:
                        if res_id in query_article["references"]:
                            recall += 1
                    recall /= len(query_article["references"])
                    self.recall_avg += recall
                except:
                    # cannot find query article
                    self.count -= 1
                    continue
        self.recall_avg /= self.count
        self.retrieval_count_avg /= self.count

    def print_stats(self):
        print(f"Recall Avg: {self.recall_avg}")
        print(f"Retrieval Count Avg: {self.retrieval_count_avg}")

if __name__ == "__main__":
    storage_file_twfp = "netzpolitik_vs_twfp.bin"
    storage_file_keywords_annotated = "netzpolitik_vs_annotated_k.bin"
    storage_file_keywords_tf_idf = "netzpolitik_vs_extracted_k.bin"
    storage_file_section_titles = "netzpolitik_vs_section_titles.bin"
    storage_file_titles = "netzpolitik_vs_titles.bin"
    es = Elasticsearch()
    parser = ParserNetzpolitik(es)
    em = EmbeddingModel()
    fe = FeatureExtraction(em, parser)

    # Index articles by:    embedding of titles
    # Query:                embedding of pre-annotated keywords
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of extracted tf-idf keywords
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_titles,
        keywords_tf_idf=True
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of extracted tf-idf keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of titles w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_first_paragraph,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of titles w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_first_paragraph,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of titles w/ section titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_section_titles,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of pre-annotated keywords
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of extracted tf-idf keywords
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_titles,
        keywords_tf_idf=True
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of extracted tf-idf keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of titles w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_first_paragraph,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of titles w/ section titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_section_titles,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of pre-annotated keywords
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of extracted tf-idf keywords
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_twfp,
        keywords_tf_idf=True
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of extracted tf-idf keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of titles w/ section titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_section_titles,
        storage_file_section_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title,
        storage_file_section_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of titles w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_first_paragraph,
        storage_file_section_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of pre-annotated keywords
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_section_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of extracted tf-idf keywords
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_section_titles,
        keywords_tf_idf=True
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of extracted tf-idf keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of pre-annotated keywords
    # Query:                embedding of pre-annotated keywords
    # Recall Avg:           0.110762
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_keywords_annotated
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of pre-annotated keywords")
    print("Query:               embedding of pre-annotated keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of extracted tf-idf keywords
    # Query:                embedding of extracted tf-idf keywords
    # Recall Avg:           0.110762
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_keywords,
        storage_file_keywords_tf_idf,
        keywords_tf_idf=True
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of extracted tf-idf keywords")
    print("Query:               embedding of extracted tf-idf keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of pre-annotated keywords
    # Query:                embedding of titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title,
        storage_file_keywords_annotated
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of pre-annotated keywords")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of extracted tf-idf keywords
    # Query:                embedding of titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title,
        storage_file_keywords_tf_idf,
        keywords_tf_idf=True
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of extracted tf-idf keywords")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of pre-annotated keywords
    # Query:                embedding of titles w/ section titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_section_titles,
        storage_file_keywords_annotated
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of pre-annotated keywords")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of extracted tf-idf keywords
    # Query:                embedding of titles w/ section titles
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_section_titles,
        storage_file_keywords_tf_idf,
        keywords_tf_idf=True
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of extracted tf-idf keywords")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of pre-annotated keywords
    # Query:                embedding of titles w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_first_paragraph,
        storage_file_keywords_annotated
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of pre-annotated keywords")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of extracted tf-idf keywords
    # Query:                embedding of titles w/ first paragraph
    exp = SemanticSearchExperiment(
        es,
        parser,
        fe.get_embedding_of_title_with_first_paragraph,
        storage_file_keywords_tf_idf
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of extracted tf-idf keywords")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")