import re
from typings import Vector
from scipy.sparse.linalg import svds
import numpy as np

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

    def get_line_separated_text_tokens(self, article):
        body = article["body"]
        tokens = body.split("\n")
        tokens = [ token for token in tokens if len(token) > 0 ]
        return tokens

    def compute_group_cosine_similarity(self, Y):
        scaled = [ np.array(vec) / np.linalg.norm(vec) for vec in Y ]
        _, s, _ = svds(scaled, k = 1)
        cos_sim_n = (s[0]**2)/len(Y)
        return cos_sim_n

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