import re
from bs4 import BeautifulSoup
from ..typings import StringList
from ..parser_interface import ParserInterface

class ParserNetzpolitik(ParserInterface):

    def __init__(self, es = None):
        self.es = es

    def get_keywords_tf_idf(self, index, article_id) -> StringList:
        # use separate request, otherwise the filter body applies for all fields
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
        body_termvector = self.es.termvectors(
            index = index,
            id = article_id,
            term_statistics = True,
            fields = ["body"],
            body = {
                "filter": {
                    "min_term_freq": 2,
                    "min_doc_freq": 5,
                    "max_num_terms": 25
                }
            }
        )
        if not title_termvector["found"] or not body_termvector["found"]:
            return []
        keywords_title = []
        if "title" in title_termvector["term_vectors"]:
            keywords_title = list(title_termvector["term_vectors"]["title"]["terms"].keys())
        keywords_body = []
        if "body" in body_termvector["term_vectors"]:
            keywords_body = list(body_termvector["term_vectors"]["body"]["terms"].keys())
        combined = list(set(keywords_title + keywords_body))
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
            while match == None:
                query = query[:-1]
                match = re.search(rf"\b{query}([\wöüäß]+)?\b", t, flags=re.IGNORECASE)
                if len(query) <= 1 and match == None:
                    return None
            return (match.group(0), match.start())

        denormalized = list(set([denorm(combined_text, keyw) for keyw in normalized if denorm(combined_text, keyw)]))
        if keep_order:
            denormalized.sort(key=lambda tupl: tupl[1])
        return [tupl[0] for tupl in denormalized]

    @staticmethod
    def parse_article(response):
        body_with_linebreaks = re.sub(r"<[\/]p>|<[\/]h[1-6]>|<br\s*[\/]?>|<[\/]figcaption>|<[\/]li>", "\n", response.text)
        soup = BeautifulSoup(body_with_linebreaks, 'lxml')
        soup_raw = BeautifulSoup(response.text, 'lxml')
        title = soup.title.text
        published = response.css('.published ::text').extract_first()
        authors = response.css('.entry-meta a[rel="author"] ::text').extract()
        categories = response.css('.entry-footer__category a[rel="tag"] ::text').extract()
        keywords = response.css('.entry-footer__tags a[rel="tag"] ::text').extract()
        references = response.css('.entry-content a ::attr(href)').extract()
        head_section = "".join([p.get_text() for p in soup.article.header.find_all('p')])
        content_section = soup.article.find_all('div', class_='entry-content')[0].get_text()
        body = head_section + content_section
        raw_body = soup_raw.article.header.find_all('p') + soup_raw.find_all('div', class_='entry-content')
        raw_body = "".join([str(tag) for tag in raw_body])
        article = {
            'url': response.url,
            'title': title,
            'published': published.lstrip()[:10].replace(".", "-"),
            'authors': authors,
            'categories': categories,
            'keywords': keywords,
            'body': body,
            'raw_body': raw_body,
            'references': references
        }
        yield article

    @staticmethod
    def get_first_paragraph(article) -> str:
        body = article["body"]
        first_p = ""
        paragraphs = ([p for p in body.split("\n") if p])[:2]
        if len(paragraphs) == 0:
            return first_p
        if len(paragraphs[0]) < 70:
            first_p += " ".join(paragraphs)
        else:
            first_p += paragraphs[0]
        return first_p.replace("  ", " ")

    @staticmethod
    def get_title_with_first_paragraph(article) -> str:
        first_p = ParserNetzpolitik.get_first_paragraph(article)
        title = ParserNetzpolitik.get_title(article)
        text = f"{title.strip()} {first_p.strip()}"
        return text.strip()

    @staticmethod
    def get_title(article) -> str:
        title = article["title"]
        return title.strip()

    @staticmethod
    def get_section_titles(article) -> StringList:
        soup = BeautifulSoup(article['raw_body'], 'lxml')
        titles = soup.find_all('h3') + soup.find_all('h2')
        titles = [title.get_text() for title in titles]
        titles = [title.strip() for title in titles if title]
        return titles
    
    @staticmethod
    def get_title_with_section_titles(article) -> str:
        title = ParserNetzpolitik.get_title(article)
        section_titles = ParserNetzpolitik.get_section_titles(article)
        combined = f"{title} {' '.join(section_titles)}"
        return combined.strip()

    @staticmethod
    def get_keywords_annotated(article) -> StringList:
        return article["keywords"]