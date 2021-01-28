from ..typings import StringList
from ..parser_interface import ParserInterface
import re

class ParserWAPO(ParserInterface):
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
    def unique_heads(entry):
        items = set()
        if type(entry) is list:
            for x in entry:
                items.add(x[0])
            return list(items)
        else:
            return entry

    @staticmethod
    def is_not_relevant(kicker: str):
        not_relevant = {
            "test",
            "opinion",
            "letters to the editor",
            "the post's view"
        }
        return (kicker.lower() in not_relevant)

# TODO
    def get_keywords(self, text:str):
        keywords = []
        return keywords

    def parse_article(self, raw):
        try:
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
            title = title.strip()
            kicker = ParserWAPO.get_first_content_by_type(raw['contents'], 'kicker')

            # ignore not relevant docs
            if "published_date" not in raw or not title or not text or ParserWAPO.is_not_relevant(kicker):
                print("irrelevant")
                return None

            data_dict = {
                "_index": "wapo",
                "_type": '_doc',
                "_id": raw['id'],
            }

            keywords = self.get_keywords(text)

            source_block = {
                "title": title,
                "offset_first_paragraph": len(first_p),
                "date": raw['published_date'],
                "kicker": kicker,
                "author": raw['author'],
                "text": text or '',
                "links": links or [],
                "url": raw['article_url'],
                "keywords": keywords

            }

            for key, val in raw.items():
                if key == key.upper():
                    print(raw['id'], ParserWAPO.unique_heads(val))
                    source_block[key] = ParserWAPO.unique_heads(val)

            data_dict['_source'] = source_block

        except Exception(e):
            return None

        return data_dict

    @staticmethod
    def get_first_paragraph_with_titles(article) -> str:
        first_p = article["text"][0:article["offset_first_paragraph"]]
        title = article["title"]
        text = title.strip() + " " + first_p.strip()
        return text