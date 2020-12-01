import re

class FeatureExtraction():

    def get_first_paragraph(self, article):
        body = article["body"]
        iter = re.finditer(r"\n", body)
        offset = next(iter).start()
        if offset < 50:
            offset = next(iter).start()
        first_p = body.replace("\n", " ")[:offset]
        first_p = first_p.replace("  ", " ")
        return first_p

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