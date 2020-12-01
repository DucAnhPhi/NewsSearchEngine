import re

class FeatureExtraction():

    @staticmethod
    def get_first_paragraph(article):
        body = article["body"]
        iter = re.finditer(r"\n", body)
        offset = next(iter).start()
        if offset < 50:
            offset = next(iter).start()
        first_p = body.replace("\n", " ")[:offset]
        first_p = first_p.replace("  ", " ")
        return first_p
