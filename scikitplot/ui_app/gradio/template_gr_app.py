# template_gr_app.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""template_gr_app."""

from scikitplot import LazyImport  # logger

# import gradio as gr
gr = LazyImport("gradio", package="gradio")

if gr:
    gr = gr.resolved

    # import spaces  # huggingface
    from scikitplot.ui_app.gradio.template_gr_b_doremi_ui import gr_bocks
    from scikitplot.ui_app.gradio.template_gr_i_doremi_ui import gr_interface

    with gr.Blocks() as app:  # noqa: SIM117
        with gr.Tabs():
            with gr.TabItem("Blocks UI"):
                # First app UI
                gr_bocks.render()
            with gr.TabItem("Interface UI"):
                # Second app UI
                gr_interface.render()

    def run_gradio(share=True):
        """Launch gradio app."""
        app.launch(share=share)

    if __name__ == "__main__":
        run_gradio(share=True)
