from typing import Optional


class AnnotationChecker:
    __task__: Optional[str] = None

    def check(self, annotation):
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def parse_from_meta(cls, meta_info: dict):
        raise NotImplementedError  # pragma: no cover
