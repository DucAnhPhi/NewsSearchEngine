import sys
sys.path.append("..")
import re
from bs4 import BeautifulSoup
from typings import StringList
from parser_interface import ParserInterface

class ParserNetzpolitik(ParserInterface):
    @staticmethod
    def parse_article(response):
        body_with_linebreaks = re.sub(r"<[\/]p>|<[\/]h[1-6]>|<br\s*[\/]?>|<[\/]figcaption>|<[\/]li>", "\n", response.text)
        soup = BeautifulSoup(body_with_linebreaks, 'lxml')
        soup_raw = BeautifulSoup(response.text, 'lxml')
        subtitle = response.css('.entry-subtitle ::text').extract_first()
        title = response.css('.entry-title ::text').extract_first()
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
            'id': response.url,
            'title': title,
            'subtitle': subtitle,
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
        paragraphs = (body.split("\n", 3))[:2]
        if len(paragraphs[0]) < 70:
            first_p += " ".join(paragraphs)
        else:
            first_p += paragraphs[0]
        return first_p.replace("  ", " ")

    @staticmethod
    def get_first_paragraph_with_titles(article) -> str:
        first_p = ParserNetzpolitik.get_first_paragraph(article)
        title = article["title"]
        subtitle = article["subtitle"]
        text = ""
        if subtitle is None:
            text += title.strip() + " " + first_p.strip()
        else:
            text += subtitle.strip() + " " + title.strip() + " " + first_p.strip()
        return text

    @staticmethod
    def get_line_separated_text_tokens(article) -> StringList:
        body = article["body"]
        tokens = body.split("\n")
        tokens = [ token for token in tokens if len(token.split()) > 0 ]
        return tokens
    
    @staticmethod
    def get_section_titles(article) -> StringList:
        soup = BeautifulSoup(article['raw_body'], 'lxml')
        titles = soup.find_all('h3') + soup.find_all('h2')
        titles = [title.get_text() for title in titles]
        return titles