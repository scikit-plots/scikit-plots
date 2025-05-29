### composer/config.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration constants used throughout the composer module.

Attributes
----------
A4_FREQ : float
    Reference frequency for note A4 in Hz.
DEFAULT_DURATION : float
    Default duration of a note in seconds.
DEFAULT_SAMPLE_RATE : int
    Default sample rate for audio synthesis (samples per second).
DEFAULT_AMPLITUDE : float
    Default peak amplitude for generated waveforms (0.0 to 1.0).
DEFAULT_BITRATE : str
    Default bitrate string for MP3 export (e.g., "192k").
DEFAULT_SOFT_CLIP_THRESHOLD : float
    Threshold level for applying soft clipping to audio signals.
MAX_INT_16BIT : int
    Maximum amplitude value for 16-bit PCM audio.
DEFAULT_AMPLITUDE_INT : int
    Default amplitude scaled to 16-bit integer range.
"""

# -------------------------------------------------------------------
# Audio synthesis constants
# -------------------------------------------------------------------

A4_FREQ: float = 440.0  #: float: Frequency of reference note A4 in Hz
DEFAULT_DURATION: float = 0.5  #: float: Default note duration in seconds
DEFAULT_SAMPLE_RATE: int = 44100  #: int: Sample rate in Hz (CD quality)
DEFAULT_AMPLITUDE: float = 1.0  #: float: Amplitude scale for soft, clear tone
DEFAULT_BITRATE: str = "192k"
DEFAULT_SOFT_CLIP_THRESHOLD: float = 0.95

MAX_INT_16BIT: int = 32767
#: int: Amplitude scale for soft, clear tone
DEFAULT_AMPLITUDE_INT: int = int(DEFAULT_AMPLITUDE * MAX_INT_16BIT)
