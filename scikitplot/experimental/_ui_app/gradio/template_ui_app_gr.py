# template_ui_app_gr.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""template_ui_app_gr."""

# from scikitplot import logger
from scikitplot._compat.optional_deps import LazyImport

__all__ = []

# import gradio as gr
gr = LazyImport("gradio", package="gradio")

if gr:
    __all__ += [
        "run_ui_app_gr",
    ]

    # import spaces  # huggingface
    from scikitplot.experimental._ui_app.gradio.template_ui_app_gr_doremi_b import ui_app_gr_doremi_b
    from scikitplot.experimental._ui_app.gradio.template_ui_app_gr_doremi_i import ui_app_gr_doremi_i

    with gr.Blocks() as ui_app_gr:  # noqa: SIM117
        with gr.Tabs():
            with gr.TabItem("UI DoReMi Blocks"):
                # First app UI
                ui_app_gr_doremi_b.render()
            with gr.TabItem("UI DoReMi Interface"):
                # Second app UI
                ui_app_gr_doremi_i.render()

    def run_ui_app_gr(share=True):
        """Launch gradio app."""
        ui_app_gr.launch(share=share)

    if __name__ == "__main__":
        run_ui_app_gr(share=True)
