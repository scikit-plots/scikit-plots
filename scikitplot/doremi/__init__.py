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
Doremi
=======

A modular Python toolkit for musical note processing, sound synthesis, and
notation handling. Supports Western and solfège notation, tone generation,
frequency mapping, waveform synthesis, and more.

Examples
--------
>>> from scikitplot import doremi
>>> doremi.compose_as_waveform()

References
----------
.. [1]: Smith, J. *Sound Synthesis for Musicians*. Audio Tech Publishing, 2021.
        https://example.com/sound-synthesis-guide.pdf

.. [2]: 3Blue1Brown. *Fourier Series*. YouTube, 2017.
        https://www.youtube.com/watch?v=spUNpyF58BY

.. [3]: muhammed celik. "How to Generate 440 Hz A(La) Note Sin Wave". Medium, May 10, 2022.
        https://celik-muhammed.medium.com/how-to-generate-440-hz-a-la-note-sin-wave-with-44-1-1e41f6ed9653
"""  # noqa: D205, D400

from .composer import *  # noqa: F403
from .config import *  # noqa: F403
from .envelopes import ENVELOPES  # noqa: F401
from .note import *  # noqa: F403
from .note_io import *  # noqa: F403
from .synthesis import *  # noqa: F403
from .waveform_playback import *  # noqa: F403
from .waveform_viz import *  # noqa: F403
