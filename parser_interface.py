import abc
from .typings import StringList

class ParserInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'get_first_paragraph_with_titles') and
            callable(subclass.get_first_paragraph_with_titles) or
            NotImplemented)

    @abc.abstractstaticmethod
    def get_first_paragraph_with_titles(article) -> str:
        raise NotImplementedError