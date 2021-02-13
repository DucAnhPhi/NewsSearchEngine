import abc
from .typings import StringList

class ParserInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'get_title') and
            callable(subclass.get_title) and
            hasattr(subclass, 'get_title_with_section_titles') and
            callable(subclass.get_title_with_section_titles) and
            hasattr(subclass, 'get_title_with_first_paragraph') and
            callable(subclass.get_title_with_first_paragraph) or
            NotImplemented)

    @abc.abstractstaticmethod
    def get_title_with_first_paragraph(article) -> str:
        raise NotImplementedError