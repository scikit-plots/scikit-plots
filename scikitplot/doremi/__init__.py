# doremi/__init__.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# composer/
# │
# ├── __init__.py
# ├── composer.py           # Main composition logic
# ├── synthesis.py          # Frequency-to-waveform + envelopes + soft clip
# ├── envelopes.py          # Envelope strategies
# ├── config.py             # Constants
# ├── io.py                 # Export (MP3, WAV), serializers (YAML, JSON)
# └── note.py               # Already exists, parsed notes

"""
Entry point for the composer module.

Exports key composition functions and utilities for external access.

Attributes
----------
generate_composition : function
    Function to generate compositions from note sequences.
note_to_sine_wave : function
    Converts musical note representations to sine waveforms.
frequency_to_sine_wave : function
    Generates a sine wave from a specified frequency.
sheet_to_note : function
    Parses sheet music notation into structured notes.

References
----------
.. [1]: muhammed celik. "How to Generate 440 Hz A(La) Note Sin Wave". Medium, May 10, 2022.
        https://celik-muhammed.medium.com/how-to-generate-440-hz-a-la-note-sin-wave-with-44-1-1e41f6ed9653
"""

from .composer import *  # noqa: F403
from .config import *  # noqa: F403
from .envelopes import ENVELOPES  # noqa: F401
from .note import *  # noqa: F403
from .note_io import *  # noqa: F403
from .synthesis import *  # noqa: F403
from .waveform_playback import *  # noqa: F403
from .waveform_viz import *  # noqa: F403
