# template_gr_app.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=no-name-in-module

"""template_gr_app."""

from scikitplot import logger
from scikitplot._compat.optional_deps import HAS_GRADIO, safe_import

logger.setLevel(logger.INFO)

if HAS_GRADIO:
    # import spaces  # huggingface
    # import gradio as gr
    gr = safe_import("gradio")

    from scikitplot.gradio.template_gr_b_doremi_ui import gr_bocks
    from scikitplot.gradio.template_gr_i_doremi_ui import gr_interface

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
