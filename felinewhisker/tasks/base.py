from typing import Optional, Callable, Type

import gradio as gr
import pandas as pd
from PIL import Image


class AnnotationChecker:
    __task__: Optional[str] = None

    def check(self, annotation):
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def parse_from_meta(cls, meta_info: dict):
        raise NotImplementedError  # pragma: no cover


ImageLoaderTyping = Callable[[str], Image.Image]


class TaskTypeRegistration:
    __task__: Optional[str]
    __cls_annotation_checker__: Type[AnnotationChecker]

    @classmethod
    def parse_annotation_checker(cls, meta_info: dict) -> AnnotationChecker:
        return cls.__cls_annotation_checker__.parse_from_meta(meta_info)

    @classmethod
    def create_annotator_ui(cls, repo, block: gr.Blocks, gr_output_state: gr.State, **kwargs) -> gr.State:
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def init_project(cls, workdir: str, task_name: str, readme_metadata: dict, **kwargs):
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def make_readme(cls, workdir: str, task_meta_info: dict, df_samples: pd.DataFrame,
                    fn_load_image: ImageLoaderTyping):
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def init_cli(cls) -> dict:
        raise NotImplementedError  # pragma: no cover
