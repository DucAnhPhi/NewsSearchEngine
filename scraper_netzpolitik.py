import scrapy
from bs4 import BeautifulSoup
import re
from scrapy.selector import Selector

class NetzpolitikScraper(scrapy.Spider):
    name="netzpolitik_spider"
    start_urls = ['https://netzpolitik.org/2020']

    def parse(self, response):
        LINK_SELECTOR = ".teaser__link.teaser__text-link.teaser__headline-link.u-uid.u-url"
        for element in response.css(LINK_SELECTOR):
            link = element.css('a ::attr(href)').extract_first()
            yield {
                'link': link
            }
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

        yield {
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