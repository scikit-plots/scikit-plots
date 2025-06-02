# template_gr_i_doremi_ui.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=no-name-in-module
# pylint: disable=line-too-long

"""template_gr_i_doremi_ui."""

from scikitplot import doremi
from scikitplot._compat.optional_deps import HAS_GRADIO, safe_import

if HAS_GRADIO:
    # import spaces  # huggingface
    # import gradio as gr
    gr = safe_import("gradio")

    ## gr.Interface - Simpler and High-Level
    gr_interface = gr.Interface(
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
        - [scikitplot.doremi](https://scikit-plots.github.io/dev/apis/scikitplot.doremi.html)  <br>
        - [ScaleFreqs.xls](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fpages.mtu.edu%2F~suits%2FScaleFreqs.xls)  <br>
        """,
    )

    def run_gr_interface_ui(share=True):
        """Launch gradio app."""
        gr_interface.launch(share=share)

    if __name__ == "__main__":
        run_gr_interface_ui(share=True)
