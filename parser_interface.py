import abc

class ParserInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'get_first_paragraph_with_titles') and
            callable(subclass.get_first_paragraph_with_titles) and
            hasattr(subclass, 'get_keywords') and
            callable(subclass.get_keywords) or
            NotImplemented)

    @abc.abstractstaticmethod
    def get_first_paragraph_with_titles(article) -> str:
        raise NotImplementedError

    @abc.abstractstaticmethod
    def get_keywords(article) -> str:
        raise NotImplementedError