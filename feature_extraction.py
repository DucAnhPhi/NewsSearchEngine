import re
from .typings import Vector, StringList
from scipy.spatial.distance import cosine
import numpy as np
from itertools import combinations
from typing import Optional

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

    def get_keywords_similarity(self, keywords: StringList):
        keywords_similarity = 2. # max cosine distance
        if len(keywords) > 1:
            embeddings = [self.embedder.encode(word) for word in keywords]
            keywords_similarity = FeatureExtraction.mean_of_pairwise_cosine_distances(embeddings)
        else:
            keywords_similarity = 0
        return keywords_similarity

    def get_embedding_of_title_with_first_paragraph(self, article) -> Optional[Vector]:
        emb = None
        if article:
            combined_text = self.parser.get_title_with_first_paragraph(article)
            emb = self.embedder.encode(combined_text)
        return emb

    def get_embedding_of_title(self, article) -> Optional[Vector]:
        emb = None
        if article:
            titles = self.parser.get_title(article)
            emb = self.embedder.encode(titles)
        return emb

    def get_embedding_of_title_with_section_titles(self, article) -> Optional[Vector]:
        emb = None
        if article:
            titles_str = self.parser.get_title(article)
            section_titles = self.parser.get_section_titles(article)
            section_titles_str = " ".join(section_titles)
            result = f"{titles_str} {section_titles_str}"
            result = result.strip()
            emb = self.embedder.encode(result)
        return emb

    def get_embedding_of_keywords(self, keywords: StringList) -> Optional[Vector]:
        if len(keywords) == 0:
            return None
        keywords_str = " ".join(keywords)
        return self.embedder.encode(keywords_str)