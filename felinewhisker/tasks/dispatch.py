from typing import Type, Dict, Callable

import pandas as pd
from PIL import Image

from .base import AnnotationChecker
from .classification import ClassificationAnnotationChecker, create_readme_for_classification

# Task Annotation checkers
_KNOWN_TASK_CHECKERS: Dict[str, Type[AnnotationChecker]] = {}


def _register_task_checker(type_: Type[AnnotationChecker]):
    _KNOWN_TASK_CHECKERS[type_.__task__] = type_


_register_task_checker(ClassificationAnnotationChecker)


def parse_annotation_checker_from_meta(meta_info: dict) -> AnnotationChecker:
    type_ = _KNOWN_TASK_CHECKERS[meta_info['task']]
    return type_.parse_from_meta(meta_info)


# README creators
_KNOWN_README_CREATOR: Dict[str, Callable] = {}


def _register_readme_creator(task_type: str, fn_creator: Callable):
    _KNOWN_README_CREATOR[task_type] = fn_creator


_register_readme_creator('classification', create_readme_for_classification)


def create_readme(f, workdir: str, task_meta_info: dict, df_samples: pd.DataFrame,
                  fn_load_image: Callable[[str], Image.Image]):
    return _KNOWN_README_CREATOR[task_meta_info['task']](
        f=f,
        workdir=workdir,
        task_meta_info=task_meta_info,
        df_samples=df_samples,
        fn_load_image=fn_load_image,
    )
