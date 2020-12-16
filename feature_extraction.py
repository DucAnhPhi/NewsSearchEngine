import re
from typings import Vector, StringList
from scipy.spatial.distance import cosine
import numpy as np
from itertools import combinations

class FeatureExtraction():

    def __init__(self, embedder=None):
        self.embedder = embedder

    def get_first_paragraph(self, article) -> str:
        body = article["body"]
        first_p = ""
        paragraphs = (body.split("\n", 3))[:2]
        if len(paragraphs[0]) < 70:
            first_p += " ".join(paragraphs)
        else:
            first_p += paragraphs[0]
        return first_p.replace("  ", " ")

    def get_first_paragraph_with_titles(self, article) -> str:
        first_p = self.get_first_paragraph(article)
        title = article["title"]
        subtitle = article["subtitle"]
        text = ""
        if subtitle is None:
            text += title.strip() + " " + first_p.strip()
        else:
            text += subtitle.strip() + " " + title.strip() + " " + first_p.strip()
        return text

    def get_line_separated_text_tokens(self, article) -> StringList:
        body = article["body"]
        tokens = body.split("\n")
        tokens = [ token for token in tokens if len(token) > 0 ]
        return tokens

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
        first_p = self.get_first_paragraph(article)
        return self.embedder.encode(first_p)

    def get_first_paragraph_with_titles_embedding(self, article) -> Vector:
        combined_text = self.get_first_paragraph_with_titles(article)
        return self.embedder.encode(combined_text)

    def get_token_embeddings(self, tokens):
        embeddings = [ self.embedder.encode(token) for token in tokens ]
        return embeddings