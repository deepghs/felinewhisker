import gradio as gr
import pandas as pd
from hbutils.string import plural_word

from ..repository import DatasetRepository


def create_squash_tab(
        repo: DatasetRepository, demo: gr.Blocks,
):
    with gr.Row(elem_id='squash_workspace'):
        with gr.Tabs():
            with gr.Tab('Data Table'):
                with gr.Row():
                    with gr.Column():
                        gr_table = gr.Dataframe()
                        gr_table_text = gr.Markdown(elem_classes='tip-text right')

            with gr.Tab('Unarchived'):
                with gr.Row():
                    with gr.Column():
                        gr_unarchived_table = gr.Dataframe()
                        gr_unarchived_text = gr.Markdown(elem_classes='tip-text right')

    def _fn_data_load():
        yield gr.update(), gr.update(), gr.update(), gr.update(), \
            gr.update(interactive=False, value='Loading'), gr.update(interactive=False),
        table = repo.read_table()
        if table is None:
            table = pd.DataFrame([])
        t_table = f'{plural_word(len(table), "archived sample")} in total.'

        records = []
        unarchived_pack_count = 0
        for _, df in repo.read_unarchived_tables():
            records.extend(df.to_dict('records'))
            unarchived_pack_count += 1
        t_unarchived = f'{plural_word(unarchived_pack_count, "unarchived packages")}, ' \
                       f'{plural_word(len(records), "sample")} in total.'

        yield table, pd.DataFrame(records), t_table, t_unarchived, \
            gr.update(interactive=True, value='Refresh'), gr.update(interactive=True)

    def _fn_squash():
        yield gr.update(interactive=False), gr.update(interactive=False, value='Squashing')
        gr.Info('Squashing ...')
        repo.squash()
        yield gr.update(interactive=True), gr.update(interactive=True, value='Squash')
        gr.Info('Squashed!')

    def _fn_init():
        return gr.update(interactive=True), gr.update(interactive=True)

    with gr.Row():
        gr_refresh = gr.Button('Refresh', interactive=False)
        gr_squash = gr.Button('Squash', variant='primary', interactive=False)

        gr_refresh.click(
            fn=_fn_data_load,
            outputs=[gr_table, gr_unarchived_table, gr_table_text, gr_unarchived_text, gr_refresh, gr_squash],
        )

        gr_squash.click(
            fn=_fn_squash,
            outputs=[gr_refresh, gr_squash]
        ).then(
            fn=_fn_data_load,
            outputs=[gr_table, gr_unarchived_table, gr_table_text, gr_unarchived_text, gr_refresh, gr_squash],
        )

    demo.load(
        fn=_fn_data_load,
        outputs=[gr_table, gr_unarchived_table, gr_table_text, gr_unarchived_text, gr_refresh, gr_squash],
    ).then(
        fn=_fn_init,
        outputs=[gr_refresh, gr_squash],
    )
