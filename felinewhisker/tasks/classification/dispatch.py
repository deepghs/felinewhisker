import logging
import re
from typing import Tuple, List, Callable

import click
import gradio as gr
import pandas as pd

from .annotation import ClassificationAnnotationChecker
from .project import create_readme_for_classification, init_project_for_classification
from .ui import create_annotator_ui_for_classification
from ..base import TaskTypeRegistration, ImageLoaderTyping


class ClassificationRegistration(TaskTypeRegistration):
    __task__ = 'classification'
    __cls_annotation_checker__ = ClassificationAnnotationChecker

    @classmethod
    def create_annotator_ui(cls, repo, block: gr.Blocks, gr_output_state: gr.State, **kwargs) -> gr.State:
        return create_annotator_ui_for_classification(
            repo=repo,
            block=block,
            gr_output_state=gr_output_state,
            **kwargs
        )

    @classmethod
    def init_project(cls, workdir: str, task_name: str, readme_metadata: dict, **kwargs):
        return init_project_for_classification(
            workdir=workdir,
            task_name=task_name,
            readme_metadata=readme_metadata,
            **kwargs
        )

    @classmethod
    def make_readme(cls, workdir: str, task_meta_info: dict, df_samples: pd.DataFrame,
                    fn_load_image: ImageLoaderTyping):
        return create_readme_for_classification(
            workdir=workdir,
            task_meta_info=task_meta_info,
            df_samples=df_samples,
            fn_load_image=fn_load_image,
        )

    @classmethod
    def init_cli(cls) -> Tuple[List[Callable], Callable]:
        options = [
            click.option('-l', '--labels', 'labels_str', type=str, required=True,
                         help='Labels for this classification task, seperated with comma.'),
        ]

        def _fn_parse(labels_str: str):
            labels = list(filter(bool, map(str.strip, re.split(r'\s*,\s*', labels_str))))
            logging.info(f'Labels of this task: {labels!r}')

            return dict(
                labels=labels,
            )

        return options, _fn_parse
