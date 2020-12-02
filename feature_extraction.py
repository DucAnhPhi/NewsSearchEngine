import re
from embedding.model import EmbeddingModel

class FeatureExtraction():

    def __init__(self, debug=False):
        if debug is False:
            self.embedder = EmbeddingModel()

    def get_first_paragraph(self, article):
        body = article["body"]
        first_p = ""
        paragraphs = (body.split("\n", 3))[:2]
        if len(paragraphs[0]) < 50:
            first_p += " ".join(paragraphs)
        else:
            first_p += paragraphs[0]
        return first_p.replace("  ", " ")

    def get_first_paragraph_with_titles(self, article):
        first_p = self.get_first_paragraph(article)
        title = article["title"]
        subtitle = article["subtitle"]
        text = ""
        if subtitle is None:
            text += title.strip() + " " + first_p.strip()
        else:
            text += subtitle.strip() + " " + title.strip() + " " + first_p.strip()
        return text

    def get_title_embedding(self, article):
        title = article["title"]
        subtitle = article["subtitle"]
        combined_title = subtitle.strip() + " " + title.strip()
        return self.embedder.encode(combined_title)

    def get_first_paragraph_embedding(self, article):
        first_p = self.get_first_paragraph(article)
        return self.embedder.encode(first_p)

    def get_first_paragraph_with_titles_embedding(self, article):
        combined_text = self.get_first_paragraph_with_titles
        return self.embedder.encode(combined_text)