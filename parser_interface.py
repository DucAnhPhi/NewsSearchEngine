import abc

class ParserInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'get_first_paragraph_with_titles') and
            callable(subclass.get_first_paragraph_with_titles) and
            hasattr(subclass, 'get_titles') and
            callable(subclass.get_titles) and
            hasattr(subclass, 'get_keywords') and
            callable(subclass.get_keywords) and
            hasattr(subclass, 'parse_article') and
            callable(subclass.parse_article) or
            NotImplemented)

    @abc.abstractstaticmethod
    def get_first_paragraph_with_titles(article) -> str:
        raise NotImplementedError

    @abc.abstractstaticmethod
    def get_titles(article) -> str:
        raise NotImplementedError

    @abc.abstractstaticmethod
    def get_keywords(article) -> str:
        raise NotImplementedError

    @abc.abstractstaticmethod
    def parse_article(response):
        raise NotImplementedError