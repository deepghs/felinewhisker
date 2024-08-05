import pathlib
from contextlib import contextmanager
from typing import Optional, ContextManager

import gradio as gr
from hbutils.string import titleize

from .annotate import create_annotation_tab
from ..datasource import BaseDataSource
from ..repository import DatasetRepository

_GLOBAL_CSS_CODE = (pathlib.Path(__file__).parent / 'global.css').read_text()


@contextmanager
def create_annotator_app(repo: DatasetRepository, datasource: BaseDataSource, author: Optional[str] = None,
                         annotation_options: Optional[dict] = None) \
        -> ContextManager[gr.Blocks]:
    with repo.write(author=author) as write_session:
        with datasource as source:
            source.set_fn_contains_id(write_session.is_id_duplicated)

            with gr.Blocks(css=_GLOBAL_CSS_CODE) as demo:
                with gr.Row(elem_id='annotation_title'):
                    gr_title = gr.HTML(
                        f'<p class="title">'
                        f'<u>{titleize(repo.meta_info["task"])}</u> - {repo.meta_info["name"]}'
                        f'</p>'
                    )
                    _ = gr_title

                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab('Annotation'):
                            create_annotation_tab(
                                repo=repo,
                                demo=demo,
                                datasource=source,
                                write_session=write_session,
                                **(annotation_options or {}),
                            )

            yield demo
