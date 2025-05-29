# template_gr_app.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""template_gr_app."""

# import spaces  # huggingface

import gradio as gr

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


if __name__ == "__main__":
    app.launch(share=True)
