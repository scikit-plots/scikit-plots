# template_ui_app_gr_doremi_b.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=unused-import
# pylint: disable=line-too-long
# pylint: disable=no-member
# ruff: noqa: F401

"""template_ui_app_gr_doremi_b."""

import atexit
import base64
import os
import shutil
import tempfile
import uuid
from datetime import datetime

# import numpy as np
# from scipy.io.wavfile import write
from scikitplot.experimental import _doremi as doremi

# from scikitplot import logger
from scikitplot._compat.optional_deps import LazyImport

__all__ = []

prefix = datetime.today().strftime("%Y-%m-%d %H:%M:%S")

## Create one shared temp folder
# TEMP_DIR = tempfile.mkdtemp()
# file_path = os.path.join(TEMP_DIR, "tone.wav")
## Clean up temp folder on shutdown
# atexit.register(lambda: shutil.rmtree(TEMP_DIR))

# Your SVG content (Download icon) fill="currentColor"
svg = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="white" width="24" height="24">
<path d="M26 24v4H6v-4H4v4a2 2 0 0 0 2 2h20a2 2 0 0 0 2-2v-4zm0-10l-1.41-1.41L17 20.17V2h-2v18.17l-7.59-7.58L6 14l10 10 10-10z"></path>
</svg>
"""
# Encode the SVG to base64
svg_base64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
img_tag = f'<img src="data:image/svg+xml;base64,{svg_base64}" width="18" height="18" style="vertical-align: middle; display:inline;">'
# img_tag = f'<span style="display:inline;">{img_tag}</span>'

# import gradio as gr
gr = LazyImport("gradio", package="gradio")

if gr:
    __all__ += [
        "run_ui_app_gr_doremi_b",
        "ui_app_gr_doremi_b",
    ]

    # import spaces  # huggingface
    ## gr.Blocks - Flexible and Modular
    with gr.Blocks() as ui_app_gr_doremi_b:
        # --- UI placeholders ---
        # ♪♪ Example UI
        gr.Markdown(
            "<h1 style='text-align: center;'>🎵 Simple Music Composer: Western and Solfège Notation</h1>"
        )
        # gr.Markdown("&nbsp;")  # Empty space to align visually
        gr.Markdown("🎶 Generate a audio wave based on composition SHEET...")

        # Rest of your layout
        with gr.Row():  # noqa: SIM117
            with gr.Column():
                # freq_input = gr.Slider(minimum=100, maximum=50000, label="Frequency (Hz)", value=440)
                # Use for MP3 from pydub import AudioSegment
                # format = gr.Radio(["wav", "mp3"], value="wav", label="Format")
                sheet_input = gr.Textbox(
                    value=doremi.SHEET.strip(),
                    label="🎼 Enter Music Composition...",
                    lines=21,  # Minimum number of visible lines
                    max_lines=40,  # Allows widget to grow up to 10 lines
                    placeholder="🎼 Enter Music Composition...",
                    show_label=True,
                    show_copy_button=True,
                    submit_btn="🎶 Generate!",
                )
                generate_btn = gr.Button("🎶 Generate!")
                # Link section
                gr.Markdown(
                    """
                ### 📘 Related Resource  <br>
                - [How to Generate 440 Hz A (La) Note Sin Wave](https://celik-muhammed.medium.com/how-to-generate-440-hz-a-la-note-sin-wave-with-44-1-1e41f6ed9653)  <br>
                - [scikitplot.experimental._doremi](https://scikit-plots.github.io/dev/apis)  <br>
                - [ScaleFreqs.xls](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fpages.mtu.edu%2F~suits%2FScaleFreqs.xls)  <br>
                """
                )

            with gr.Column():
                # output = gr.Textbox(label="Response")

                # The audio play button often appears active but doesn't respond immediately.
                # It requires a small delay to function correctly.
                # Play generated audio feed: filepath or (sample_rate, audio array)
                # audio_output = gr.Audio(label="Player: Generated Audio", type="filepath")
                # https://github.com/gradio-app/gradio/issues/8177
                audio_output = gr.Audio(
                    label="🔊 Attention: Please wait for soundbars..."
                )  # , autoplay=True, streaming=True, format="mp3" | "bytes",

                # Always visible accordion with instructions, collapsed by default
                with gr.Accordion("🔔 Trouble playing the audio?", open=False):
                    # Text with embedded icon
                    gr.Markdown(
                        f"""
                        👉 If playback doesn't display Soundbars:
                        - {img_tag} Click the **Download** button (top-right of the **Audio Playback**).
                        - 🎬 **view** option, if any: Open audio with your browsers's default player.
                        - 🔽 **download** option: On mobile, downloading may offer smoother playback.
                        - 📲 Open the file in your preferred media player.
                        """
                    )

                gr.Markdown(
                    "Choose how the sound's volume gradually starts and ends to avoid abrupt noises and clicks."
                )
                envelope_choice = gr.Radio(
                    choices=list(doremi.ENVELOPES),
                    label="Envelope Type",
                    value="hann",  # Default selection
                )

        # --- Add/Call Action ---
        # Gradio dynamically adds event methods like .click() at runtime.

        # Called when the user changes the textbox value
        # freq_input.change(
        #     fnf= ...,                 # your function
        #     inputs = [...],           # component(s) to pass as input
        #     outputs = [...],          # component(s) to update as output
        # )

        # Trigger when Enter key is pressed or the submit button in the textbox
        sheet_input.submit(
            # return filepath or (sample_rate, audio array)
            # lambda *a, **kw: doremi.save_waveform(doremi.compose_as_waveform(*a, **kw), file_path=file_path)
            fn=lambda *a, **kw: (
                doremi.DEFAULT_SAMPLE_RATE,
                doremi.compose_as_waveform(*a, **kw),
            ),
            inputs=[
                sheet_input,
                envelope_choice,
            ],
            outputs=audio_output,  # one output by fn
        )

        # Trigger an action when the user changes a gr.Radio selection
        envelope_choice.change(
            # return filepath or (sample_rate, audio array)
            # lambda *a, **kw: doremi.save_waveform(doremi.compose_as_waveform(*a, **kw), file_path=file_path)
            fn=lambda *a, **kw: (
                doremi.DEFAULT_SAMPLE_RATE,
                doremi.compose_as_waveform(*a, **kw),
            ),
            inputs=[
                sheet_input,
                envelope_choice,
            ],
            outputs=audio_output,  # one output by fn
        )

        # Trigger when the button is clicked
        generate_btn.click(
            # return filepath or (sample_rate, audio array)
            # lambda *a, **kw: doremi.save_waveform(doremi.compose_as_waveform(*a, **kw), file_path=file_path)
            fn=lambda *a, **kw: (
                doremi.DEFAULT_SAMPLE_RATE,
                doremi.compose_as_waveform(*a, **kw),
            ),
            inputs=[
                sheet_input,
                envelope_choice,
            ],
            outputs=audio_output,  # one output by fn
        )

    def run_ui_app_gr_doremi_b(share=True):
        """Launch gradio app."""
        ui_app_gr_doremi_b.launch(share=share)

    if __name__ == "__main__":
        run_ui_app_gr_doremi_b(share=True)
