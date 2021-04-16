from ..typings import StringList
from ..parser_interface import ParserInterface
import re

class ParserWAPO(ParserInterface):

    def __init__(self, es = None):
        self.es = es

    def get_keywords_tf_idf(self, index, article_id) -> StringList:
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
        if not text_termvector["found"]:
            return []
        keywords_title = []
        keywords_text = []
        if "title" in title_termvector["term_vectors"]:
            keywords_title = list(title_termvector["term_vectors"]["title"]["terms"].keys())
        if "text" in text_termvector["term_vectors"]:
            keywords_text = list(text_termvector["term_vectors"]["text"]["terms"].keys())
        combined = list(set(keywords_title + keywords_text))
        return combined

    def get_keywords_tf_idf_denormalized(self, index, article_id, title, text, keep_order = True) -> StringList:
        normalized = self.get_keywords_tf_idf(index, article_id)

        if len(normalized) == 0:
            return normalized

        combined_text = f"{title if title is not None else ''} {text if text is not None else ''}".strip()
        if not combined_text:
            return []

        def denorm(t, kw):
            query = kw
            match = re.search(rf"\b{query}([\wöüäß]+)?\b", t, flags=re.IGNORECASE)
            while match is None:
                query = query[:-1]
                match = re.search(rf"\b{query}([\wöüäß]+)?\b", t, flags=re.IGNORECASE)
                if len(query) <= 1 and match is None:
                    return None
            return (match.group(0), match.start())

        denormalized = list(set([denorm(combined_text, keyw) for keyw in normalized if denorm(combined_text, keyw)]))
        if keep_order:
            denormalized.sort(key=lambda tupl: tupl[1])
        return [tupl[0] for tupl in denormalized]

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
    def get_section_titles(jsarr):
        titles = [c["content"] for c in jsarr if c is not None and c["type"] == "sanitized_html" and "subtype" in c and (c["subtype"] == "subhead" or c["subtype"] == "sublabel") and "content" in c and c["content"] is not None]
        return titles

    @staticmethod
    def is_not_relevant(kicker: str):
        is_not_relevant = False
        if kicker:
            not_relevant = {
                "test",
                #"opinion",
                #"letters to the editor",
                #"the post's view"
            }
            is_not_relevant = kicker.lower() in not_relevant
        return is_not_relevant

    @staticmethod
    def parse_article(raw, ignore=True):
        text = ParserWAPO.get_all_content_by_type(raw['contents'], 'sanitized_html')
        section_titles = ParserWAPO.get_section_titles(raw["contents"])
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
        if ignore and ((not text) or ParserWAPO.is_not_relevant(kicker)):
            return None
        source_block = {
            "title": title or '',
            "section_titles": section_titles,
            "offset_first_paragraph": len(first_p),
            "date": ParserWAPO.get_first_content_by_type(raw['contents'], 'date'),
            "kicker": kicker,
            "author": raw['author'],
            "text": text or '',
            "links": links or [],
            "url": raw['article_url']
        }
        return source_block

    @staticmethod
    def get_title(article) -> str:
        title = article["title"]
        return title.strip()

    @staticmethod
    def get_title_with_section_titles(article) -> str:
        title = ParserWAPO.get_title(article)
        section_titles = " ".join(article["section_titles"])
        combined = f"{title} {section_titles}"
        return combined.strip()

    @staticmethod
    def get_title_with_first_paragraph(article) -> str:
        first_p = article["text"][0:article["offset_first_paragraph"]]
        title = ParserWAPO.get_title(article)
        text = f"{title} {first_p.strip()}"
        return text.strip()