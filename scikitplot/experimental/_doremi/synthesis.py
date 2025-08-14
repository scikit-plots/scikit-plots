### composer/synthesis.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# pylint: disable=broad-exception-caught
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

"""
Audio synthesis utilities including sine wave generation
and soft clipping functionality.

- Sine wave generation with optional envelopes (e.g. Hann, soft)

Functions
---------
soft_clip(waveform, threshold)
    Applies soft clipping to a waveform.
frequency_to_sine_wave(frequency, duration, amplitude, sample_rate, envelope)
    Generates a sine wave with a given frequency and envelope.
"""  # noqa: D205

from typing import Union

import numpy as np

from ... import logger
from ..._compat.python import lru_cache
from .config import DEFAULT_AMPLITUDE, DEFAULT_DURATION, DEFAULT_SAMPLE_RATE
from .envelopes import get_envelope
from .note import parse_note_token

__all__ = [
    "frequency_to_sine_wave",
    "note_to_sine_wave",
]

# -------------------------------------------------------------------
# Generate Sine Wave
# -------------------------------------------------------------------


@lru_cache(maxsize=128)
def frequency_to_sine_wave(
    frequency: float,
    duration: float = DEFAULT_DURATION,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    envelope: "Union[str, callable[[np.ndarray, float], np.ndarray], None]" = None,
) -> np.ndarray:
    r"""
    Generate a sine waveform at a given frequency with optional amplitude envelope (fade-in/out).

    The sine wave is defined as:

    .. math::
        y(t) = A \cdot \sin(2\pi f t) \cdot E(t)

    where:
      - \( f \) is the frequency in Hz
      - \( A \) is the amplitude
      - \( E(t) \) is the envelope over time \( t \)

    Parameters
    ----------
    frequency : float
        Frequency of the sine wave in Hz.
    duration : float, optional
        Duration of the waveform in seconds. Default is 0.5.
    amplitude : float, optional
        Peak amplitude of the waveform. Default is 1.0.
    sample_rate : int, optional
        Sampling rate in Hz. Default is 44100.
    envelope : str or callable or None, optional
        Amplitude envelope shaping function over time. Options:
        - 'hann' (default)
        - 'soft'
        - 'triangular'
        - Callable with signature (t: np.ndarray, duration: float) -> np.ndarray
        - If `None`, no envelope is applied (flat amplitude).

    Returns
    -------
    np.ndarray
        The generated waveform as a float32 NumPy array.

    Notes
    -----
    Envelopes smooth the attack/decay to avoid clicks in audio:

    - **Hann envelope**:
      .. math:: E(t) = 0.5 \cdot (1 - \cos(2 \pi t / T))

    - **Soft envelope**:
      .. math:: E(t) = \sin(\pi t / T)

    - **Triangular envelope**:
      .. math:: E(t) = 1 - |(2t/T) - 1|

    Examples
    --------
    >>> frequency_to_sine_wave(440.0, duration=1.0)
    >>> frequency_to_sine_wave(261.63, envelope="soft")
    >>> frequency_to_sine_wave(523.25, amplitude=0.7, envelope="triangular")
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    env = 1.0 if envelope is None else get_envelope(t, duration, envelope)
    waveform = amplitude * np.sin(2 * np.pi * frequency * t) * env
    return waveform.astype(np.float32)


@lru_cache(maxsize=128)
def note_to_sine_wave(  # noqa: D417
    note: str,
    octave: int = 4,
    duration: float = DEFAULT_DURATION,
    amplitude: float = DEFAULT_AMPLITUDE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    envelope: "Union[str, callable[[np.ndarray, float], np.ndarray], None]" = None,
) -> np.ndarray:
    """
    Generate a sine waveform for a given musical note.

    This function supports both Western (e.g., 'C4') and Solfège (e.g., 'Do4') notation,
    with optional fallback values for octave and duration.

    Parameters
    ----------
    note : str
        Musical note string, e.g., "C4", "Do4", "C#4-0.5", "Do-1".
        If duration or octave is missing, fallback values are appended.
    octave : int, default=4
        Fallback octave if not included in the note string.
    duration : float, default=0.5
        Fallback duration in seconds if not provided in the note string.
    amplitude : float, default=1.0
        Amplitude of the generated waveform.
    sample_rate : int, default=44100
        Number of audio samples per second.
    envelope : str or callable, optional
        Envelope to shape amplitude over time.
        Choose from: 'hann', 'soft', 'triangular', or provide a custom function.

    Returns
    -------
    np.ndarray
        Array of audio samples as a sine wave.

    Raises
    ------
    ValueError
        If the note cannot be parsed into a valid frequency or duration.

    Examples
    --------
    >>> note_to_sine_wave("C4-0.5")
    >>> note_to_sine_wave("Do", octave=4, duration=1.0)
    """
    # Auto-fill missing octave or duration
    if note and not any(char.isdigit() for char in note):
        note = f"{note}{octave}"
    if "-" not in note:
        note = f"{note}-{duration}"

    try:
        token = parse_note_token(note)
        # freq = token.frequency2()
        freq = token.frequency()
        dur = token.duration
    except Exception as e:
        raise ValueError(f"Failed to parse note '{note}': {e}") from e

    return frequency_to_sine_wave(
        frequency=freq,
        duration=dur,
        amplitude=amplitude,
        sample_rate=sample_rate,
        envelope=envelope,
    )


# -------------------------------------------------------------------
# CLI for Quick Testing
# -------------------------------------------------------------------

# %%
if __name__ == "__main__":
    import soundfile as sf  # type: ignore[reportMissingImports]

    tone = note_to_sine_wave("Do", duration=0.5)
    sf.write("Do4.wav", tone, samplerate=DEFAULT_SAMPLE_RATE)
    logger.info("song.wav saved.")
