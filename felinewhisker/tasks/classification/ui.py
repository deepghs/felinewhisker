import gradio as gr

from ...utils import emoji_image_file

_DEFAULT_HOTKEY_MAPS = [
    *(str(i) for i in range(1, 10)),
    *(chr(ord('a') + i) for i in range(26))
]
_DEFAULT = object()

_HOTKEY_EMOJIS = {
    *(str(i) for i in range(1, 10)),
    *(chr(ord('a') + i) for i in range(26))
}


def create_annotator_ui_for_classification(repo, block: gr.Blocks, gr_output_state: gr.State, hotkey_maps=_DEFAULT):
    from ...repository import DatasetRepository
    repo: DatasetRepository

    labels = repo.meta_info['labels']
    hotkey_maps = _DEFAULT_HOTKEY_MAPS if hotkey_maps is _DEFAULT else hotkey_maps

    with gr.Row(elem_id='annotation_workspace'):
        gr_position_id = gr.State(value=-1)
        gr_sample_id = gr.State(value=None)

        with gr.Column():
            gr_sample = gr.Image(type='pil', label='', elem_classes='limit-height')

        with gr.Column(elem_classes='limit-height'):
            with gr.Row():
                with gr.Column():
                    gr_annotation = gr.State(value=object())

                    btn_label_ids = [f'btn_label_{label}' for i, label in enumerate(labels)]
                    btns = [gr.Button(
                        value=(
                            f'({hotkey_maps[i].upper()}) {label}'
                            if hotkey_maps[i] not in _HOTKEY_EMOJIS else label
                        ),
                        elem_id=btn_id,
                        elem_classes='btn-label',
                        interactive=False,
                        icon=(
                            emoji_image_file(f':keycap_{hotkey_maps[i].upper()}:')
                            if hotkey_maps[i] in _HOTKEY_EMOJIS else None
                        ),
                    ) for i, (label, btn_id) in enumerate(zip(labels, btn_label_ids))]

                    def _annotation_transition(annotation, triggered_state):
                        new_state = triggered_state if (annotation != triggered_state) else None
                        return new_state

                    for i, label in enumerate(labels):
                        gr_button = btns[i]
                        gr_button.click(
                            fn=_annotation_transition,
                            inputs=[gr_annotation, gr.State(value=label)],
                            outputs=[gr_annotation],
                        )

            with gr.Row():
                gr_unannotate_button = gr.Button(
                    value='Unannotate',
                    elem_id='btn_unannoate_label',
                    icon=emoji_image_file(':no_entry:'),
                    interactive=False,
                )
                gr_unannotate_button.click(
                    fn=_annotation_transition,
                    inputs=[gr_annotation, gr.State(value=None)],
                    outputs=[gr_annotation],
                )

                def _annotation_changed(current_position_id, state):
                    new_btns = [
                        gr.update(
                            elem_classes='btn-label btn-selected' if state and label == state else 'btn-label',
                            interactive=True,
                        ) for i, (label, btn_id) in enumerate(zip(labels, btn_label_ids))
                    ]
                    unannotate_button = gr.update(interactive=True if state else False)
                    if state:
                        state_html = f'<p>Current Sample: #{current_position_id}</p>' \
                                     f'<p>Annotated: <b>{state}</b></p>'
                    else:
                        state_html = f'<p>Current Sample: #{current_position_id}</p>' \
                                     f'<p>Unannotated</p>'
                    return state_html, unannotate_button, *new_btns

                gr_annotation_text = gr.HTML(elem_classes='tip-text right')
                gr_annotation.change(
                    fn=_annotation_changed,
                    inputs=[gr_position_id, gr_annotation],
                    outputs=[gr_annotation_text, gr_unannotate_button, *btns],
                ).then(
                    fn=lambda _id, _annotation: (_id, _annotation),
                    inputs=[gr_sample_id, gr_annotation],
                    outputs=[gr_output_state],
                )

            js_elses = " else ".join([
                f"if (event.key === {str(hotkey_maps[i])!r}) {{ document.getElementById({btn_id!r}).click(); }}"
                for i, (label, btn_id) in enumerate(zip(labels, btn_label_ids))
            ])
            js_hotkeys = f"""
                    function () {{
                        document.addEventListener('keydown', function(event) {{
                            if (event.key === 'Escape') {{
                                document.getElementById('btn_unannoate_label').click();
                            }} else {{
                                {js_elses}
                            }}
                        }});
                    }}
                    """

            block.load(None, js=js_hotkeys)

        gr_state_input = gr.State(value=None)

        def _state_change(state):
            position_id, sample_id, image, annotation = state
            return position_id, sample_id, image, annotation

        gr_state_input.change(
            fn=_state_change,
            inputs=[gr_state_input],
            outputs=[gr_position_id, gr_sample_id, gr_sample, gr_annotation]
        ).then(
            fn=_annotation_changed,
            inputs=[gr_position_id, gr_annotation],
            outputs=[gr_annotation_text, gr_unannotate_button, *btns],
        )

    return gr_state_input
