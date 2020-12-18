import scrapy
from scrapy.selector import Selector
from typings import StringList
from parser_netzpolitik import ParserNetzpolitik

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

def get_start_urls() -> StringList:
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
            yield scrapy.Request(link, ParserNetzpolitik.parse_article)