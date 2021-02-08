from ..typings import StringList
from ..parser_interface import ParserInterface
import re

class ParserWAPO(ParserInterface):

    def __init__(self, es = None):
        self.es = es

    def get_keywords(self, index, article_id):
        # use separate request, otherwise the filter body applies for both fields
        title_termvector = self.es.termvectors(
            index = index,
            id = article_id,
            term_statistics = True,
            fields = ["title"],
            body = {
                "filter": {
                    "min_term_freq": 1,
                    "min_doc_freq": 1,
                    "max_num_terms": 3
                }
            }
        )
        text_termvector = self.es.termvectors(
            index = index,
            id = article_id,
            term_statistics = True,
            fields = ["text"],
            body = {
                "filter": {
                    "min_term_freq": 2,
                    "min_doc_freq": 5,
                    "max_num_terms": 25
                }
            }
        )
        keywords_title = list(title_termvector["term_vectors"]["title"]["terms"].keys())
        keywords_text = list(text_termvector["term_vectors"]["text"]["terms"].keys())
        combined = list(set(keywords_title + keywords_text))
        return combined

    @staticmethod
    def get_first_content_by_type(jsarr, t):
        for block in jsarr:
            if block is not None and block["type"] == t:
                return block["content"]
        return None

    @staticmethod
    def get_first_paragraph(jsarr):
        first_p = ""
        for block in jsarr:
            if block is not None and block["type"] == "sanitized_html":
                first_p += f"{block['content']} "
                if block["subtype"] == "paragraph" and len(first_p) > 50:
                    return first_p.strip()
        return ""

    @staticmethod
    def get_all_content_by_type(jsarr, t, field="content"):
        strings = [c[field] for c in jsarr if c is not None and c["type"] == t and field in c and c[field] is not None]
        if strings:
            return " ".join(strings)
        else:
            return None

    @staticmethod
    def is_not_relevant(kicker: str):
        is_not_relevant = False
        if kicker:
            not_relevant = {
                "test",
                "opinion",
                "letters to the editor",
                "the post's view"
            }
            is_not_relevant = kicker.lower() in not_relevant
        return is_not_relevant

    @staticmethod
    def parse_article(raw):
        text = ParserWAPO.get_all_content_by_type(raw['contents'], 'sanitized_html')
        first_p = ParserWAPO.get_first_paragraph(raw['contents'])
        first_p = re.sub('<.*?>', ' ', first_p)
        first_p = first_p.strip()
        links = []
        if text:
            links = re.findall('href="([^"]*)"', text)
            text = re.sub('<.*?>', ' ', text)
            text = text.strip()
        title = ParserWAPO.get_all_content_by_type(raw['contents'], 'title')
        if title:
            title.strip()
        kicker = ParserWAPO.get_first_content_by_type(raw['contents'], 'kicker')
        # ignore not relevant docs
        if ("published_date" not in raw) or (not title) or (not text) or ParserWAPO.is_not_relevant(kicker):
            return None
        source_block = {
            "title": title,
            "offset_first_paragraph": len(first_p),
            "date": raw['published_date'],
            "kicker": kicker,
            "author": raw['author'],
            "text": text or '',
            "links": links or [],
            "url": raw['article_url']
        }
        return source_block

    @staticmethod
    def get_first_paragraph_with_titles(article) -> str:
        article = article["_source"]
        first_p = article["text"][0:article["offset_first_paragraph"]]
        title = article["title"]
        text = title.strip() + " " + first_p.strip()
        return text