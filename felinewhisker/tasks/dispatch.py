from typing import Type, Dict

from .base import AnnotationChecker
from .classification import ClassificationAnnotationChecker

_KNOWN_TASK_CHECKERS: Dict[str, Type[AnnotationChecker]] = {}


def _register_task_checker(type_: Type[AnnotationChecker]):
    _KNOWN_TASK_CHECKERS[type_.__task__] = type_


_register_task_checker(ClassificationAnnotationChecker)


def parse_annotation_checker_from_meta(meta_info: dict) -> AnnotationChecker:
    type_ = _KNOWN_TASK_CHECKERS[meta_info['task']]
    return type_.parse_from_meta(meta_info)
