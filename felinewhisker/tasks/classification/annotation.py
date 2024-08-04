from typing import List

from ..base import AnnotationChecker


class ClassificationAnnotationChecker(AnnotationChecker):
    __task__ = 'classification'

    def __init__(self, labels: List[str]):
        self._labels = list(labels)
        self._label_set = set(labels)

    def check(self, annotation):
        if isinstance(annotation, str) and annotation in self._label_set:
            pass
        else:
            raise ValueError(f'Invalid annotation {annotation!r} for {self}.')

    def __repr__(self):
        return f'<{self.__class__.__name__} labels: {self._labels}>'

    @classmethod
    def parse_from_meta(cls, meta_info: dict) -> 'ClassificationAnnotationChecker':
        return cls(labels=meta_info['labels'])
