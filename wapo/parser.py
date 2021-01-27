from ..typings import StringList
from ..parser_interface import ParserInterface

class ParserWAPO(ParserInterface):
    @staticmethod
    def get_first_paragraph(article) -> str:
        return article["text"][0:len(article["offset_first_paragraph"])]

    @staticmethod
    def get_first_paragraph_with_titles(article) -> str:
        first_p = ParserWAPO.get_first_paragraph(article)
        title = article["title"]
        text = title.strip() + " " + first_p.strip()
        return text

    @staticmethod
    def get_keywords(article) -> StringList:
        return article['keywords']