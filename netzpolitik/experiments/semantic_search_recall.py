import os.path
import pathlib
import json
from elasticsearch import Elasticsearch
from ...vector_storage import VectorStorage
from ...netzpolitik.parser import ParserNetzpolitik
from ...feature_extraction import FeatureExtraction
from ...embedding.model import EmbeddingModel
from ...typings import NearestNeighborList

class SemanticSearchExperiment():
    def __init__(self, index_emb_func, query_emb_func, storage_file):
        self.es = Elasticsearch()
        self.index = "netzpolitik"
        self.count = 0
        self.recall_avg = 0.
        self.retrieval_count_avg = 0.

        # init vector storage
        data_location = f"{os.path.abspath(os.path.join(__file__ , os.pardir, os.pardir, os.pardir))}/data"
        storage_location = f"{data_location}/{storage_file}"
        if os.path.isfile(storage_location):
            print("Loading vector storage from file...\n")
            self.vs = VectorStorage(storage_location)
        else:
            print("Initialize vector storage.\n")
            self.vs = VectorStorage()
            print("Add items from file...\n")
            with open(f"{data_location}/netzpolitik.jsonl", 'r', encoding="utf-8") as data_file:
                emb_batch: VectorList = []
                id_batch: StringList = []

                for line in data_file:
                    article = json.loads(line)
                    emb = index_emb_func(article)
                    if emb == None:
                        continue
                    emb_batch.append(emb)
                    id_batch.append(article["id"])

                    if len(emb_batch) is 1000:
                        self.vs.add_items(emb_batch, id_batch)
                        emb_batch = []
                        id_batch = []

                if len(emb_batch) is not 0:
                    self.vs.add_items(emb_batch, id_batch)
            self.vs.save(storage_location)

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
                    query = query_emb_func(query_article)
                    if query == None:
                        self.count -= 1
                        continue
                    nearest_n: NearestNeighborList = self.vs.get_k_nearest([query],100)
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
    storage_file_twfp = "storage_titles_w_first_p.bin"
    storage_file_keywords = "storage_keywords.bin"
    storage_file_section_titles = "storage_section_titles.bin"
    storage_file_titles = "storage_titles.bin"
    em = EmbeddingModel()
    fe = FeatureExtraction(em, ParserNetzpolitik())

    # Index articles by:    embedding of titles
    # Query:                embedding of keywords
    # Recall Avg:           
    # Retrieval Count Avg:  
    exp = SemanticSearchExperiment(
        fe.get_titles_embedding,
        fe.get_keywords_embedding,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of titles w/ first paragraph
    # Recall Avg:           0.10652200049654036
    # Retrieval Count Avg:  100.0
    exp = SemanticSearchExperiment(
        fe.get_first_paragraph_with_titles_embedding,
        fe.get_first_paragraph_with_titles_embedding,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of titles
    # Recall Avg:           0.06588950547460022
    # Retrieval Count Avg:  100.0
    exp = SemanticSearchExperiment(
        fe.get_titles_embedding,
        fe.get_titles_embedding,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of titles w/ first paragraph
    # Recall Avg:           0.05631176486970014
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        fe.get_titles_embedding,
        fe.get_first_paragraph_with_titles_embedding,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of titles w/ section titles
    # Recall Avg:           0.06049119822426727
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        fe.get_titles_embedding,
        fe.get_titles_w_section_titles_embedding,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles
    # Query:                embedding of keywords
    # Recall Avg:           
    # Retrieval Count Avg:  
    exp = SemanticSearchExperiment(
        fe.get_titles_embedding,
        fe.get_keywords_embedding,
        storage_file_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles")
    print("Query:               embedding of keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of titles w/ first paragraph
    # Recall Avg:           0.10652200049654036
    # Retrieval Count Avg:  100.0
    exp = SemanticSearchExperiment(
        fe.get_first_paragraph_with_titles_embedding,
        fe.get_first_paragraph_with_titles_embedding,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of titles
    # Recall Avg:           0.54156
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        fe.get_first_paragraph_with_titles_embedding,
        fe.get_titles_embedding,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of titles w/ section titles
    # Recall Avg:           0.05715172283821507
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        fe.get_first_paragraph_with_titles_embedding,
        fe.get_titles_w_section_titles_embedding,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ first paragraph
    # Query:                embedding of keywords
    # Recall Avg:           0.027038
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        fe.get_first_paragraph_with_titles_embedding,
        fe.get_keywords_embedding,
        storage_file_twfp
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ first paragraph")
    print("Query:               embedding of keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of titles w/ section titles
    # Recall Avg:           0.068766
    # Retrieval Count Avg:  100.0
    exp = SemanticSearchExperiment(
        fe.get_titles_w_section_titles_embedding,
        fe.get_titles_w_section_titles_embedding,
        storage_file_section_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of titles
    # Recall Avg:           0.063814
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        fe.get_titles_w_section_titles_embedding,
        fe.get_titles_embedding,
        storage_file_section_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of titles w/ first paragraph
    # Recall Avg:           
    # Retrieval Count Avg:  
    exp = SemanticSearchExperiment(
        fe.get_titles_w_section_titles_embedding,
        fe.get_first_paragraph_with_titles_embedding,
        storage_file_section_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of titles w/ section titles
    # Query:                embedding of keywords
    # Recall Avg:           
    # Retrieval Count Avg:  
    exp = SemanticSearchExperiment(
        fe.get_titles_w_section_titles_embedding,
        fe.get_keywords_embedding,
        storage_file_section_titles
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of titles w/ section titles")
    print("Query:               embedding of keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of keywords
    # Query:                embedding of keywords
    # Recall Avg:           0.110762
    # Retrieval Count Avg:  100
    exp = SemanticSearchExperiment(
        fe.get_keywords_embedding,
        fe.get_keywords_embedding,
        storage_file_keywords
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of keywords")
    print("Query:               embedding of keywords")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of keywords
    # Query:                embedding of titles
    # Recall Avg:           
    # Retrieval Count Avg:  
    exp = SemanticSearchExperiment(
        fe.get_keywords_embedding,
        fe.get_titles_embedding,
        storage_file_keywords
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of keywords")
    print("Query:               embedding of titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of keywords
    # Query:                embedding of titles w/ section titles
    # Recall Avg:           
    # Retrieval Count Avg:  
    exp = SemanticSearchExperiment(
        fe.get_keywords_embedding,
        fe.get_titles_w_section_titles_embedding,
        storage_file_keywords
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of keywords")
    print("Query:               embedding of titles w/ section titles")
    exp.print_stats()
    print("----------------------------------------------------------------\n")

    # Index articles by:    embedding of keywords
    # Query:                embedding of titles w/ first paragraph
    # Recall Avg:           
    # Retrieval Count Avg:  
    exp = SemanticSearchExperiment(
        fe.get_keywords_embedding,
        fe.get_first_paragraph_with_titles_embedding,
        storage_file_keywords
    )
    print("----------------------------------------------------------------")
    print("Index articles by:   embedding of keywords")
    print("Query:               embedding of titles w/ first paragraph")
    exp.print_stats()
    print("----------------------------------------------------------------\n")