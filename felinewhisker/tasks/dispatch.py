from typing import Dict, Callable, Type

import gradio as gr
import pandas as pd
from PIL import Image

from .base import AnnotationChecker, TaskTypeRegistration
from .classification import ClassificationRegistration

_KNOWN_TASK_TYPES: Dict[str, Type[TaskTypeRegistration]] = {}


def register_task_type(reg_cls: Type[TaskTypeRegistration]):
    assert reg_cls.__task__, f'Task type should not be empty - {reg_cls!r}.'
    _KNOWN_TASK_TYPES[reg_cls.__task__] = reg_cls


register_task_type(ClassificationRegistration)


def parse_annotation_checker_from_meta(meta_info: dict) -> AnnotationChecker:
    return _KNOWN_TASK_TYPES[meta_info['task']].parse_annotation_checker(
        meta_info=meta_info
    )


def create_readme(workdir: str, task_meta_info: dict, df_samples: pd.DataFrame,
                  fn_load_image: Callable[[str], Image.Image]):
    return _KNOWN_TASK_TYPES[task_meta_info['task']].make_readme(
        workdir=workdir,
        task_meta_info=task_meta_info,
        df_samples=df_samples,
        fn_load_image=fn_load_image,
    )


def init_project(task_type: str, workdir: str, task_name: str, readme_metadata: dict, **kwargs):
    return _KNOWN_TASK_TYPES[task_type].init_project(
        workdir=workdir,
        task_name=task_name,
        readme_metadata=readme_metadata,
        **kwargs
    )


def create_annotator_ui(repo, block: gr.Blocks, gr_output_state: gr.State, **kwargs) -> gr.State:
    from ..repository import DatasetRepository
    repo: DatasetRepository

    return _KNOWN_TASK_TYPES[repo.meta_info['task']].create_ui(
        repo=repo,
        block=block,
        gr_output_state=gr_output_state,
        **kwargs
    )
