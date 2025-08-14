# note_io.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=broad-exception-caught

"""waveform_playback."""

import os

import numpy as np
from scipy.io import wavfile

from ... import logger
from .config import DEFAULT_SAMPLE_RATE

__all__ = [
    "listen_waveform",
    "play_waveform",
]


def listen_waveform(
    waveform: np.ndarray,
    sample_rate: int | None = None,
):
    """
    Play a waveform using the sounddevice library.

    Parameters
    ----------
    waveform : np.ndarray
        Audio waveform to play.
    sample_rate : int, optional
        Sampling rate in Hz (default is 44100).

    Returns
    -------
    None

    Notes
    -----
    This function uses the `sounddevice` module to play audio directly from a NumPy array.

    Examples
    --------
    >>> wave = generate_note_wave("C", 4)
    >>> listen_waveform(wave)
    """
    import sounddevice as sd  # type: ignore[reportMissingImports]

    if waveform.size > 0:
        sd.play(waveform, sample_rate)
        sd.wait()


def play_waveform(
    music: np.ndarray,
    rate: int | None = None,
    file_path: str | None = None,
    blocking: bool = True,
    backend: str = "auto",
    save_generated: bool = False,
) -> None:
    """
    Play audio from a NumPy array using either IPython (for Jupyter) or sounddevice.

    Parameters
    ----------
    music : np.ndarray
        A 1D (mono) or 2D (stereo) NumPy array containing audio waveform data.
        Expected dtype is float32 or float64 with values in [-1.0, 1.0].
    rate : int
        Sampling rate of the audio in Hz (e.g., 44100).
    file_path : str, optional
        If this is a path to an existing `.wav` file, it will be played.
        If `music` is provided and `save_generated=True`, the array is saved to this file.
    blocking : bool, optional
        Whether to block until playback finishes (only applies to sounddevice).
    backend : {'auto', 'jupyter', 'sounddevice'}, optional
        Audio playback backend:
        - 'auto': Use IPython if in Jupyter, else fallback to sounddevice.
        - 'jupyter': Force IPython.display.Audio.
        - 'sounddevice': Force sounddevice playback.
    save_generated : bool, optional
        If True and `file_path` is provided, saves `music` to `file_path` before playing.

    Returns
    -------
    dict
        Playback result info with fields:
        - 'status': 'success' or 'error'
        - 'backend': backend used
        - 'blocking': whether playback was blocking
        - 'error': error message if playback failed
        - 'source': file or music

    Raises
    ------
    ValueError
        If the backend is not one of 'auto', 'jupyter', or 'sounddevice'
        or inputs are misconfigured.
    Exception
        If audio playback fails.

    Examples
    --------
    >>> import numpy as np
    >>> duration = 1.0  # seconds
    >>> rate = 44100
    >>> t = np.linspace(0, duration, int(rate * duration), endpoint=False)
    >>> sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 tone
    >>> play_waveform(sine_wave, rate)

    Automatically picks best option:

    >>> play_waveform(music, rate=doremi.DEFAULT_SAMPLE_RATE)

    Force Jupyter output:

    >>> play_waveform(music, rate=doremi.DEFAULT_SAMPLE_RATE, backend="jupyter")

    Force sounddevice:

    >>> play_waveform(music, rate=doremi.DEFAULT_SAMPLE_RATE, backend="sounddevice")

    Notes
    -----
    🔧 Requires either IPython (for Jupyter playback) or the `sounddevice` package.
    To install sounddevice and its backend::

        # Install PortAudio library (needed for sounddevice to work on Linux)
        sudo apt-get install libportaudio2 libportaudiocpp0 portaudio19-dev
        pip install sounddevice

    If you're in Docker or WSL/Linux without sound, You must either:
    - Mount PulseAudio socket or use --device /dev/snd and libasound2
    - Or avoid using sounddevice and stick to IPython.display.Audio or exporting .wav files
    """
    result = {
        "status": "success",
        "backend": None,
        "blocking": blocking,
        "error": None,
        "source": "music" if music is not None else "file",
    }

    # Detect if running in Jupyter
    def in_jupyter():
        try:
            from IPython import get_ipython

            return get_ipython() is not None
        except ImportError:
            return False

    def use_ipython(data, rate, filename=None):
        from IPython.display import Audio, display

        if filename:
            display(Audio(filename))
        display(Audio(data=data, rate=rate))

    def use_sounddevice(data, rate):
        import sounddevice as sd

        try:
            # Assume `music` is a NumPy array with audio data
            sd.play(data=data, samplerate=rate)  # device=0
            if blocking:
                sd.wait()  # Wait until playback is done
        except Exception as e:
            logger.error(sd.query_devices())
            raise RuntimeError(f"Sounddevice playback failed: {e}") from e

    try:
        if file_path and os.path.isfile(file_path):
            # Load and play from file
            rate, data = wavfile.read(file_path)
            if backend == "jupyter" or (backend == "auto" and in_jupyter()):
                use_ipython(data, rate)
                result["backend"] = "jupyter"
            else:
                use_sounddevice(data, rate)
                result["backend"] = "sounddevice"
            return result

        if music is None:
            raise ValueError(
                "Must provide either `music` or a valid `file_path` to an existing file."
            )

        # Validate music array
        if not isinstance(music, np.ndarray):
            raise TypeError("`music` must be a NumPy array.")
        if music.ndim > 2:  # noqa: PLR2004
            raise ValueError("`music` must be a 1D or 2D NumPy array.")
        if music.dtype not in [np.float32, np.float64]:
            logger.warning(
                "⚠️ Warning: Expected float32 or float64 in [-1.0, 1.0] for proper audio scaling."
            )

        # Optionally save music
        if file_path and save_generated:
            scaled = np.int16(np.clip(music, -1.0, 1.0) * 32767)
            wavfile.write(file_path, rate, scaled)
            result["source"] = f"generated+saved to '{file_path}'"

        rate = rate or DEFAULT_SAMPLE_RATE
        # Playback generated audio
        if backend in {"jupyter", "auto"} and in_jupyter():
            use_ipython(music, rate)
            result["backend"] = "jupyter"
        elif backend in {"sounddevice", "auto"}:
            use_sounddevice(music, rate)
            result["backend"] = "sounddevice"
        else:
            raise ValueError(f"Unknown backend: {backend}")

    except (RuntimeError, ValueError, Exception) as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"❌ Playback failed: {e}")

    return result
