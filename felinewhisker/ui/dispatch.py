import pathlib
from contextlib import contextmanager
from typing import Optional, ContextManager

import gradio as gr
from hbutils.string import titleize, plural_word

from ..datasource import BaseDataSource, ImageItem
from ..repository import DatasetRepository
from ..tasks import create_ui_for_annotator

_GLOBAL_CSS_CODE = (pathlib.Path(__file__).parent / 'global.css').read_text()
_HOTKEY_JS_CODE = (pathlib.Path(__file__).parent / 'hotkeys.js').read_text()


@contextmanager
def create_annotator_app(repo: DatasetRepository, datasource: BaseDataSource, author: Optional[str] = None) \
        -> ContextManager[gr.Blocks]:
    with repo.write(author=author) as write_session:
        with datasource as source:
            source.set_fn_contains_id(write_session.is_id_duplicated)
            data_iterator = iter(source)

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
                            gr_state_output = gr.State(value=None)
                            gr_position_id = gr.State(value=-1)
                            gr_id_list = gr.State(value=[])

                            with gr.Row():
                                gr_state_input = create_ui_for_annotator(
                                    repo=repo,
                                    block=demo,
                                    gr_output_state=gr_state_output,
                                )

                            with gr.Row():
                                gr_prev = gr.Button(value='Prev (←)', elem_id='left-button')
                                gr_next = gr.Button(value='Next (→)', elem_id='right-button')
                                gr_save = gr.Button(value='Save (Ctrl+S)', elem_id='save-button')

                            def _fn_prev(idx, ids):
                                if idx <= 0:
                                    raise gr.Error('This is the first image, no previous sample.')
                                idx -= 1
                                return idx, ids

                            gr_prev.click(
                                fn=_fn_prev,
                                inputs=[gr_position_id, gr_id_list],
                                outputs=[gr_position_id, gr_id_list],
                            )

                            def _fn_next(idx, ids):
                                idx += 1
                                if idx >= len(ids):
                                    try:
                                        gr.Info(f'Loading image #{idx} ...')
                                        item: ImageItem = next(data_iterator)
                                    except StopIteration:
                                        raise gr.Error('No more images in the data source, '
                                                       'you have met the end.')
                                    sample_id = item.id
                                    with item.make_file(force_reencode=True) as image_file:
                                        write_session.add(
                                            id_=item.id,
                                            image_file=image_file,
                                            annotation=item.annotation,
                                        )
                                    ids.append(sample_id)

                                return idx, ids

                            gr_next.click(
                                fn=_fn_next,
                                inputs=[gr_position_id, gr_id_list],
                                outputs=[gr_position_id, gr_id_list],
                            )

                            def _ch_change(state):
                                id_, annotation = state
                                write_session[id_] = annotation
                                print(f'{id_} --> {state!r}')

                            gr_state_output.change(
                                fn=_ch_change,
                                inputs=[gr_state_output],
                            )

                            def _fn_index_change(idx, ids):
                                sample_id = ids[idx]
                                annotation = write_session[sample_id]
                                image_file = write_session.get_image_path(sample_id)
                                return idx, sample_id, image_file, annotation

                            gr_position_id.change(
                                fn=_fn_index_change,
                                inputs=[gr_position_id, gr_id_list],
                                outputs=[gr_state_input],
                            )

                            def _fn_save():
                                save_count = write_session.get_annotated_count()
                                gr.Info(f'Saving {plural_word(save_count, "annotated sample")} ...')
                                write_session.save()
                                gr.Info(f'{plural_word(save_count, "sample")} saved!')

                            gr_save.click(
                                fn=_fn_save,
                            )

                            demo.load(None, js=_HOTKEY_JS_CODE)

            yield demo
