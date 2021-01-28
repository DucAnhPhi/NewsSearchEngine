from ..typings import StringList
from ..parser_interface import ParserInterface
import re

class ParserWAPO(ParserInterface):
    @staticmethod
    def get_first_content_by_type(jsarr, t):
        for block in jsarr:
            if block is not None and block['type'] == t:
                return block['content']
        return None

    @staticmethod
    def get_all_content_by_type(jsarr, t, field='content'):
        strings = [c[field] for c in jsarr if c is not None and c['type'] == t and field in c and c[field] is not None]
        if strings:
            return ' '.join(strings)
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
    def get_keywords(text:str):
        keywords = []
        return keywords

    def parse_article(raw):
        try:
            text = ParserWAPO.get_all_content_by_type(js['contents'], 'sanitized_html')
            first_p = ParserWAPO.get_first_content_by_type(js['contents'], 'sanitized_html')
            links = []
            if text:
                links = re.findall('href="([^"]*)"', text)
                text = re.sub('<.*?>', ' ', text)
            title = ParserWAPO.get_all_content_by_type(js['contents'], 'title')
            kicker = ParserWAPO.get_first_content_by_type(js['contents'], 'kicker')

            # ignore not relevant docs
            if "published_date" not in js or not title.strip() or not text.strip() or ParserWAPO.is_not_relevant(kicker):
                return None

            data_dict = {
                "_index": args.index_name,
                "_type": '_doc',
                "_id": js['id'],
            }

            keywords = ParserWAPO.get_keywords(text)

            source_block = {
                "title": title,
                "offset_first_paragraph": len(first_p),
                "date": js['published_date'],
                "kicker": kicker,
                "author": js['author'],
                "text": text or '',
                "links": links or [],
                "url": js['article_url'],
                "keywords": keywords

            }

            for key, val in js.items():
                if key == key.upper():
                    print(js['id'], ParserWAPO.unique_heads(val))
                    source_block[key] = ParserWAPO.unique_heads(val)

            data_dict['_source'] = source_block

        except Exception:
            # print(json.dumps(js,sort_keys=True, indent=4))
            traceback.print_exc(file=sys.stdout)
            quit()

        return data_dict

    @staticmethod
    def get_first_paragraph(article) -> str:
        return article["text"][0:len(article["offset_first_paragraph"])]

    @staticmethod
    def get_first_paragraph_with_titles(article) -> str:
        first_p = ParserWAPO.get_first_paragraph(article)
        title = article["title"]
        text = title.strip() + " " + first_p.strip()
        return text