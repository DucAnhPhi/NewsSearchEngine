import re
from typings import Vector, StringList
from scipy.spatial.distance import cosine
import numpy as np
from itertools import combinations
from parser_netzpolitik import ParserNetzpolitik

class FeatureExtraction():

    def __init__(self, embedder=None):
        self.embedder = embedder

    def mean_of_pairwise_cosine_distances(self, embeddings) -> float:
        # all combinations of 2 rows, ignoring order and no repeated elements
        coms = combinations(embeddings, 2)
        dists = np.array([cosine(pair[0], pair[1]) for pair in coms])
        mean_dist = np.mean(dists)
        return mean_dist

    def get_title_embedding(self, article) -> Vector:
        title = article["title"]
        subtitle = article["subtitle"]
        combined_title = subtitle.strip() + " " + title.strip()
        return self.embedder.encode(combined_title)

    def get_first_paragraph_embedding(self, article) -> Vector:
        first_p = ParserNetzpolitik.get_first_paragraph(article)
        return self.embedder.encode(first_p)

    def get_first_paragraph_with_titles_embedding(self, article) -> Vector:
        combined_text = ParserNetzpolitik.get_first_paragraph_with_titles(article)
        return self.embedder.encode(combined_text)

    def get_token_embeddings(self, tokens):
        embeddings = [ self.embedder.encode(token) for token in tokens ]
        return embeddings

    def get_semantic_specifity(self, article):
        text_tokens = ParserNetzpolitik.get_section_titles(article)
        token_embeddings = self.get_token_embeddings(text_tokens)
        semantic_specifity = self.mean_of_pairwise_cosine_distances(token_embeddings)
        return semantic_specifity