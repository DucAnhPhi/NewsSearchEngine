import re
from embedding.model import EmbeddingModel
from typings import Vector

class FeatureExtraction():

    def __init__(self, debug=False):
        if debug is False:
            self.embedder = EmbeddingModel()

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

    def get_title_embedding(self, article) -> Vector:
        title = article["title"]
        subtitle = article["subtitle"]
        combined_title = subtitle.strip() + " " + title.strip()
        return self.embedder.encode(combined_title)

    def get_first_paragraph_embedding(self, article) -> Vector:
        first_p = self.get_first_paragraph(article)
        return self.embedder.encode(first_p)

    def get_first_paragraph_with_titles_embedding(self, article) -> Vector:
        combined_text = self.get_first_paragraph_with_titles
        return self.embedder.encode(combined_text)