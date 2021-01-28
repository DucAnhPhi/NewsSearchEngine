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

    def get_first_paragraph_with_titles_embedding(self, article, parse_articles = False) -> Optional[Vector]:
        emb = None
        if parse_articles:
            article = self.parser.parse_article(article)
        if article != None:
            combined_text = self.parser.get_first_paragraph_with_titles(article)
            emb = self.embedder.encode(combined_text)
        return emb

    def get_titles_embedding(self, article) -> Vector:
        titles = self.parser.get_titles(article)
        return self.embedder.encode(titles)

    def get_titles_w_section_titles_embedding(self, article) -> Vector:
        titles_str = self.parser.get_titles(article)
        section_titles = self.parser.get_section_titles(article)
        section_titles_str = " ".join(section_titles)
        result = f"{titles_str} {section_titles_str}"
        result = result.strip()
        return self.embedder.encode(result)

    def get_keywords_similarity(self, article):
        text_tokens = article["keywords"]
        keywords_similarity = 2 # max cosine distance
        if len(text_tokens) > 1:
            token_embeddings = [self.embedder.encode(token) for token in text_tokens]
            keywords_similarity = FeatureExtraction.mean_of_pairwise_cosine_distances(token_embeddings)
        else:
            keywords_similarity = 0
        return keywords_similarity

    def get_keywords_embedding(self, article) -> Optional[Vector]:
        keywords = article["keywords"]
        if len(keywords) == 0:
            return None
        keywords_str = " ".join(keywords)
        return self.embedder.encode(keywords_str)