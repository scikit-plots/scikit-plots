# template_ui_app_gr_doremi_i.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=line-too-long

"""template_ui_app_gr_doremi_i."""

# import numpy as np
# from scipy.io.wavfile import write
from scikitplot.experimental import _doremi as doremi

# from scikitplot import logger
from scikitplot._compat.optional_deps import LazyImport

__all__ = []

# import gradio as gr
gr = LazyImport("gradio", package="gradio")

if gr:
    __all__ += [
        "run_ui_app_gr_doremi_i",
        "ui_app_gr_doremi_i",
    ]

    # import spaces  # huggingface
    ## gr.Interface - Simpler and High-Level
    ui_app_gr_doremi_i = gr.Interface(
        title="🎵 Simple Music Composer: Western and Solfège Notation",
        description="🎶 Generate a audio wave based on composition SHEET...",
        # Play generated audio (INPUTS: SHEET OUTPUTS: np.array)
        fn=lambda *a, **kw: (
            doremi.DEFAULT_SAMPLE_RATE,
            doremi.compose_as_waveform(*a, **kw),
        ),
        inputs=[
            # gr.Dropdown(notes, type="index"),
            # gr.Slider(4, 6, step=1),
            gr.Textbox(
                value=doremi.SHEET.strip(),
                label="🎼 Enter Music Composition...",
                lines=21,  # Minimum number of visible lines
                max_lines=40,  # Allows widget to grow up to 10 lines
                placeholder="🎼 Enter Music Composition...",
                show_label=True,
                show_copy_button=True,
                submit_btn="🎶 Generate!",
            ),
        ],
        # (sample_rate, waveform.astype(np.float32))  # ✅ this is required syntax
        # outputs = "audio",  # component or layout
        outputs=gr.Audio(label="🔊 Attention: Please wait for soundbars..."),
        article="""
        ### 📘 Related Resource  <br>
        - [How to Generate 440 Hz A (La) Note Sin Wave](https://celik-muhammed.medium.com/how-to-generate-440-hz-a-la-note-sin-wave-with-44-1-1e41f6ed9653)  <br>
        - [scikitplot.experimental._doremi](https://scikit-plots.github.io/dev/apis)  <br>
        - [ScaleFreqs.xls](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fpages.mtu.edu%2F~suits%2FScaleFreqs.xls)  <br>
        """,
    )

    def run_ui_app_gr_doremi_i(share=True):
        """Launch gradio app."""
        ui_app_gr_doremi_i.launch(share=share)

    if __name__ == "__main__":
        run_ui_app_gr_doremi_i(share=True)
