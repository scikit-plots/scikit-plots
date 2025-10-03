# doremi/__init__.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# ruff: noqa: F405
# Flake8: noqa: F401,F403
# type: ignore[]
# pylint: disable=import-error,import-self,undefined-all-variable

"""
Doremi
=======

A modular Python toolkit for musical note processing, sound synthesis, and
notation handling. Supports Western and solfège notation, tone generation,
frequency mapping, waveform synthesis, and more.

See [1]_, [2]_, and [3]_ for model details.

Examples
--------
>>> from scikitplot.experimental import _doremi as doremi
>>> doremi.compose_as_waveform()


References
----------
.. [1] `Smith, J. (2021).
   *Sound Synthesis for Musicians*.
   Audio Tech Publishing. https://example.com/sound-synthesis-guide.pdf
   <https://example.com/sound-synthesis-guide.pdf>`_

.. [2] `3Blue1Brown. (2017).
   *Fourier Series*.
   YouTube. https://www.youtube.com/watch?v=spUNpyF58BY
   <https://www.youtube.com/watch?v=spUNpyF58BY>`_

.. [3] `Çelik, M. (2022, May 9).
   "How to generate 440 Hz A(LA) Note Sin wave with 44.1"
   Medium. https://celik-muhammed.medium.com/how-to-generate-440-hz-a-la-note-sin-wave-with-44-1-1e41f6ed9653
   <https://celik-muhammed.medium.com/how-to-generate-440-hz-a-la-note-sin-wave-with-44-1-1e41f6ed9653>`_
"""  # noqa: D205, D400

# composer/
# │
# ├── __init__.py
# ├── composer.py           # Main composition logic
# ├── synthesis.py          # Frequency-to-waveform + envelopes + soft clip
# ├── envelopes.py          # Envelope strategies
# ├── config.py             # Constants
# ├── io.py                 # Export (MP3, WAV), serializers (YAML, JSON)
# └── note.py               # Already exists, parsed notes

from ..._testing._pytesttester import PytestTester  # Pytest testing
from .composer import *  # noqa: F403
from .config import *  # noqa: F403
from .envelopes import ENVELOPES  # noqa: F401
from .note import *  # noqa: F403
from .note_io import *  # noqa: F403
from .synthesis import *  # noqa: F403
from .waveform_playback import *  # noqa: F403
from .waveform_viz import *  # noqa: F403

test = PytestTester(__name__)
del PytestTester

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = [
    "A4_FREQ",
    "DEFAULT_AMPLITUDE",
    "DEFAULT_AMPLITUDE_INT",
    "DEFAULT_BITRATE",
    "DEFAULT_DURATION",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SOFT_CLIP_THRESHOLD",
    "ENVELOPES",
    "MAX_INT_16BIT",
    "SHEET",
    "compose_as_waveform",
    "composer",
    "config",
    "envelopes",
    "export_notes_to_files",
    "export_sheet",
    "frequency_to_sine_wave",
    "listen_waveform",
    "note",
    "note_io",
    "note_to_sine_wave",
    "note_utils",
    "play_waveform",
    "plot_waveform",
    "save_waveform",
    "save_waveform_as_mp3",
    "serialize_sheet",
    "sheet_add_frequency",
    "sheet_converter",
    "sheet_to_note",
    "synthesis",
    "test",
    "waveform_playback",
    "waveform_viz",
]
