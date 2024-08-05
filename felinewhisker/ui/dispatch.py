import logging
import os
import pathlib
from contextlib import contextmanager
from typing import Optional, ContextManager, Callable, Any

import gradio as gr
from hbutils.string import titleize
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client
from huggingface_hub.errors import LocalTokenNotFoundError

from .annotate import create_annotation_tab
from .squash import create_squash_tab
from ..datasource import BaseDataSource
from ..repository import DatasetRepository

_GLOBAL_CSS_CODE = (pathlib.Path(__file__).parent / 'global.css').read_text()


@contextmanager
def create_annotator_app(
        repo: DatasetRepository, datasource: BaseDataSource, author: Optional[str] = None,
        fn_annotate_assist: Optional[Callable[[str], Any]] = None,
        annotation_options: Optional[dict] = None
) -> ContextManager[gr.Blocks]:
    hf_client = get_hf_client()

    if not author:
        try:
            info = hf_client.whoami()
            author = info['name']
        except LocalTokenNotFoundError:
            logging.warning('Huggingface auth failed, no author name used, this session will run as guest.')
            author = None

    with TemporaryDirectory(prefix='felinewhisker_') as td_state, \
            repo.write(author=author) as write_session, datasource as source:
        state_file = os.path.join(td_state, 'state.json')
        source.set_fn_contains_id(write_session.is_id_duplicated)

        with gr.Blocks(css=_GLOBAL_CSS_CODE) as demo:
            with gr.Row(elem_id='annotation_title'):
                with gr.Column():
                    gr_title = gr.HTML(
                        f'<p class="title">'
                        f'<u>{titleize(repo.meta_info["task"])}</u> - {repo.meta_info["name"]}'
                        f'</p>'
                    )
                    _ = gr_title

                    if author:
                        gr_subtitle = gr.HTML(
                            f'<p class="subtitle">'
                            f'Hello, <u>@{author}</u>!'
                            f'</p>'
                        )
                    else:
                        gr_subtitle = gr.HTML(
                            f'<p class="subtitle negative">'
                            f'(<b>Warning</b>: Running in Guest Mode)'
                            f'</p>'
                        )
                    _ = gr_subtitle

            with gr.Row():
                with gr.Tabs():
                    with gr.Tab('Annotation'):
                        create_annotation_tab(
                            repo=repo,
                            demo=demo,
                            datasource=source,
                            write_session=write_session,
                            state_file=state_file,
                            fn_annotate_assist=fn_annotate_assist,
                            **(annotation_options or {}),
                        )

                    with gr.Tab('Squash'):
                        create_squash_tab(
                            repo=repo,
                            demo=demo,
                        )

        yield demo
