import gradio as gr
import pandas as pd

from ..repository import DatasetRepository


def create_squash_tab(
        repo: DatasetRepository, demo: gr.Blocks,
):
    with gr.Row():
        with gr.Tabs():
            with gr.Tab('Data Table'):
                gr_table = gr.Dataframe()

            with gr.Tab('Unarchived'):
                gr_unarchived_table = gr.Dataframe()
                pass

    def _fn_data_load():
        table = repo.read_table()
        if table is None:
            table = pd.DataFrame([])

        records = []
        for _, df in repo.read_unarchived_tables():
            records.extend(df.to_dict('records'))

        return table, pd.DataFrame(records)

    def _fn_squash():
        gr.Info('Squashing ...')
        repo.squash()
        gr.Info('Squashed!')

    with gr.Row():
        gr_refresh = gr.Button('Refresh')
        gr_squash = gr.Button('Squash', variant='primary')

        gr_refresh.click(
            fn=_fn_data_load,
            outputs=[gr_table, gr_unarchived_table],
        )

        gr_squash.click(
            fn=_fn_squash,
        ).then(
            fn=_fn_data_load,
            outputs=[gr_table, gr_unarchived_table],
        )

    demo.load(
        fn=_fn_data_load,
        outputs=[gr_table, gr_unarchived_table],
    )
