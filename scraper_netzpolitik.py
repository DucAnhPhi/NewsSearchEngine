import scrapy
import re
import json
from bs4 import BeautifulSoup
from scrapy.selector import Selector

baseurl = "https://netzpolitik.org/"
num_pages = {
    '2010': 77,
    '2011': 65,
    '2012': 71,
    '2013': 103,
    '2014': 105,
    '2015': 110,
    '2016': 81,
    '2017': 69,
    '2018': 57,
    '2019': 61,
    '2020': 55
}

def get_start_urls():
    urls = []
    for year in num_pages.keys():
        urls.append(baseurl + year)
        urls = urls + [baseurl + year + '/page/' + str(page) for page in range(2,num_pages[year]+1)]
    return urls


class NetzpolitikScraper(scrapy.Spider):

    name="netzpolitik_spider"

    custom_settings = {
        'FEEDS': {
            # store huge amount of data in JSON Lines format. See: https://jsonlines.org/
            'data/netzpolitik.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8'
            }
        }
    }

    start_urls = get_start_urls()

    def parse(self, response):
        LINK_SELECTOR = ".teaser__link.teaser__text-link.teaser__headline-link.u-uid.u-url"
        for element in response.css(LINK_SELECTOR):
            link = element.css('a ::attr(href)').extract_first()
            yield scrapy.Request(link, self.parse_article)

    def parse_article(self, response):
        body_with_linebreaks = re.sub(r"<[\/]p>|<[\/]h[1-6]>|<br\s*[\/]?>|<[\/]figcaption>|<[\/]li>", "\n", response.text)
        soup = BeautifulSoup(body_with_linebreaks, 'lxml')

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

        article = {
            'id': response.url,
            'title': title,
            'subtitle': subtitle,
            'published': published.lstrip()[:10].replace(".", "-"),
            'authors': authors,
            'categories': categories,
            'keywords': keywords,
            'body': body,
            'references': references
        }

        yield article