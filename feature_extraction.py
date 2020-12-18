import re
from .typings import Vector, StringList
from scipy.spatial.distance import cosine
import numpy as np
from itertools import combinations

class FeatureExtraction():

    def __init__(self, embedder, parser):
        self.embedder = embedder
        self.parser = parser

    @staticmethod
    def mean_of_pairwise_cosine_distances(embeddings) -> float:
        # all combinations of 2 rows, ignoring order and no repeated elements
        coms = combinations(embeddings, 2)
        dists = np.array([cosine(pair[0], pair[1]) for pair in coms])
        mean_dist = np.mean(dists)
        return mean_dist

    def get_first_paragraph_with_titles_embedding(self, article) -> Vector:
        combined_text = self.parser.get_first_paragraph_with_titles(article)
        return self.embedder.encode(combined_text)

    def get_semantic_specifity(self, article):
        text_tokens = self.parser.get_section_titles(article)
        token_embeddings = [self.embedder.encode(token) for token in text_tokens]
        semantic_specifity = FeatureExtraction.mean_of_pairwise_cosine_distances(token_embeddings)
        return semantic_specifity