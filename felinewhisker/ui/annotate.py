import datetime
import html
import inspect
import json
import os.path
import pathlib
from typing import Iterable, Callable, Optional, Any

import gradio as gr
from hbutils.string import plural_word

from ..datasource import ImageItem
from ..repository import DatasetRepository, WriterSession
from ..tasks import create_annotator_ui
from ..utils import emoji_image_file

_HOTKEY_JS_CODE = (pathlib.Path(__file__).parent / 'hotkeys.js').read_text()


def get_fn_signature(func: Callable) -> str:
    if hasattr(func, '__name__'):
        func_name = func.__name__
    else:
        func_name = '<fn_anonymous>'

    sig = inspect.signature(func)
    parameters = sig.parameters
    param_names = [param for param in parameters]

    formatted_signature = f"{func_name}({', '.join(param_names)})"
    return formatted_signature


def create_annotation_tab(
        repo: DatasetRepository, demo: gr.Blocks,
        datasource: Iterable[ImageItem], write_session: WriterSession, state_file: str,
        fn_annotate_assist: Optional[Callable[[str], Any]] = None, **kwargs
):
    data_iterator = iter(datasource)

    gr_state_output = gr.State(value=None)
    gr_position_id = gr.State(value=-1)
    gr_max_length = gr.State(value=None)
    gr_id_list = gr.State(value=[])

    def _fn_state_save(position_id, max_length, id_list):
        with open(state_file, 'w') as f:
            json.dump({
                'position_id': position_id,
                'max_length': max_length,
                'id_list': id_list,
            }, f, indent=4, sort_keys=True)

    gr_position_id.change(
        fn=_fn_state_save,
        inputs=[gr_position_id, gr_max_length, gr_id_list],
    )
    gr_max_length.change(
        fn=_fn_state_save,
        inputs=[gr_position_id, gr_max_length, gr_id_list],
    )
    gr_id_list.change(
        fn=_fn_state_save,
        inputs=[gr_position_id, gr_max_length, gr_id_list],
    )

    def _fn_load_state():
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                return state['position_id'], state['max_length'], state['id_list']
        else:
            return -1, None, []

    demo.load(
        fn=_fn_load_state,
        outputs=[gr_position_id, gr_max_length, gr_id_list],
    )

    with gr.Row():
        gr_state_input = create_annotator_ui(
            repo=repo,
            block=demo,
            gr_output_state=gr_state_output,
            **kwargs
        )

    with gr.Row():
        gr_prev = gr.Button(
            value='Prev',
            elem_id='left-button',
            icon=emoji_image_file(':left_arrow:'),
            interactive=False,
        )
        gr_next = gr.Button(
            value='Next',
            elem_id='right-button',
            icon=emoji_image_file(':right_arrow:'),
        )
        gr_save = gr.Button(
            value='Save (Ctrl+S)',
            elem_id='save-button',
            icon=emoji_image_file(':floppy_disk:'),
            interactive=False,
        )

    with gr.Row(elem_classes='bottom-state'):
        with gr.Column(scale=2):
            gr.HTML(
                f"<p>Session: <u>{html.escape(write_session.session_token)}</u></p>",
                elem_classes='bottom-state-session'
            )
        with gr.Column(scale=1):
            if fn_annotate_assist:
                gr.HTML(
                    f'<p>Annotation Assistent: <u>{html.escape(get_fn_signature(fn_annotate_assist))}</u></p>',
                    elem_classes='bottom-state-assistant'
                )
        with gr.Column(scale=1):
            gr_save_state = gr.HTML(elem_classes='bottom-state-save-time')

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

    def _fn_next(idx, ids, max_length):
        origin_idx = idx
        idx += 1
        if idx >= len(ids):
            try:
                gr.Info(f'Loading image #{idx} ...')
                item: ImageItem = next(data_iterator)
            except StopIteration:
                gr.Warning('No more images in the data source, '
                           'you have met the end.')
                idx = origin_idx
                max_length = len(ids)
            else:
                sample_id = item.id
                with item.make_file(force_reencode=True) as image_file:
                    write_session.add(
                        id_=item.id,
                        image_file=image_file,
                        annotation=item.annotation,
                    )
                ids.append(sample_id)

        return idx, ids, max_length

    gr_next.click(
        fn=_fn_next,
        inputs=[gr_position_id, gr_id_list, gr_max_length],
        outputs=[gr_position_id, gr_id_list, gr_max_length],
    )

    def _ch_change(state):
        id_, annotation = state
        write_session[id_] = annotation

    gr_state_output.change(
        fn=_ch_change,
        inputs=[gr_state_output],
    )

    def _fn_index_change(idx, ids, max_length):
        sample_id = ids[idx]
        annotation = write_session[sample_id]
        image_file = write_session.get_image_path(sample_id)

        if fn_annotate_assist and annotation is None:
            gr.Info(f'Annotating sample #{idx} by assistant ...')
            annotation = fn_annotate_assist(image_file)
            if annotation is not None:
                write_session[sample_id] = annotation
                gr.Info(f'Sample #{idx} auto-annotated by assistant - {annotation!r}.')
            else:
                gr.Warning(f'No recommendation for sample #{idx}.')

        return (idx, sample_id, image_file, annotation), \
            gr.update(interactive=idx > 0), \
            gr.update(interactive=max_length is None or idx < max_length - 1), \
            gr.update(interactive=True)

    gr_position_id.change(
        fn=_fn_index_change,
        inputs=[gr_position_id, gr_id_list, gr_max_length],
        outputs=[gr_state_input, gr_prev, gr_next, gr_save],
    )

    def _fn_save():
        yield gr.update(interactive=False, value='Saving'), gr.update()
        save_count = write_session.get_annotated_count()
        gr.Info(f'Saving {plural_word(save_count, "annotated sample")} ...')
        write_session.save()
        yield gr.update(interactive=True, value='Save (Ctrl+S)'), \
            f'<p>Last Saved at: {datetime.datetime.now()}</p>'
        gr.Info(f'{plural_word(save_count, "sample")} saved!')

    gr_save.click(
        fn=_fn_save,
        outputs=[gr_save, gr_save_state],
    )

    demo.load(None, js=_HOTKEY_JS_CODE)
