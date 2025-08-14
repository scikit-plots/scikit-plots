### composer/io.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=import-outside-toplevel
# pylint: disable=no-else-return
# pylint: disable=unused-variable
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments

"""
I/O utilities for saving generated waveforms to WAV or MP3 formats.

Functions
---------
save_waveform(waveform, filename, samplerate)
    Saves waveform data to a WAV file.
save_waveform_as_mp3(waveform, filename, bitrate, sample_rate)
    Saves waveform data to an MP3 file.
"""

import os
from pathlib import Path

import numpy as np

from ... import logger
from .config import DEFAULT_AMPLITUDE_INT, DEFAULT_BITRATE, DEFAULT_SAMPLE_RATE
from .note_utils import preprocess_waveform

__all__ = [
    "save_waveform",
    "save_waveform_as_mp3",
]


def save_waveform(
    waveform: np.ndarray,
    file_path: str = "output.wav",
    ext: str | None = None,
    sample_rate: int | None = DEFAULT_SAMPLE_RATE,
    backend: str | None = None,
    dtype: str = "float32",
    normalize: bool = True,
    stereo_out: bool = True,
    **kwargs,
) -> str:
    """
    Save waveform to an audio file using specified or auto-selected backend.

    Parameters
    ----------
    waveform : np.ndarray | array-like
        Audio array, 1D or 2D. Will be reshaped, normalized, and cast.
    file_path : str, default="output.wav"
        Output file_path.
    ext : str or None, optional
        File extension override (e.g., 'wav', 'flac'). Inferred from file_path if None.
    sample_rate : int, default=44100
        Sample rate in Hz. Defaults to 44100 if None.
    backend : str or None, optional
        Audio backend to use: 'scipy', 'soundfile', or 'scitools'.
        Auto-selected based on extension if None.
    dtype : str, default='int16'
        Output audio dtype: 'int16' or 'float32'.
    normalize : bool, default=True
        Whether to normalize waveform amplitude to [-1,1].
    stereo_out : bool
        Convert mono to stereo by duplication if True.
    **kwargs
        Additional arguments passed to backend writers.

    Returns
    -------
    str
        The file path of the saved audio.

    Raises
    ------
    ValueError
        If backend unsupported or invalid extension for backend.

    Notes
    -----
    - 1D mono and 2D multi-channel supported.
    - Channel dimension must be second in shape (samples, channels).
    - Audio will be passed through `preprocess_audio` before saving.
    - Mono audio should be saved as a 1D array with shape (samples,).
    - Audio file formats like WAV expect mono data as a single stream of samples, 1D array.
    - Most audio libraries (e.g., soundfile, scipy.io.wavfile, librosa.output.write_wav)
      accept mono data as 1D arrays.
    - Stereo/multi-channel audio must be saved as a 2D array with shape (samples, channels).
    - The order of dimensions is important, channels must be the second dimension for
      correct playback.
    - Saving with the wrong shape (e.g., (channels, samples)) will cause audio distortion or
      channel mix-up.

    Examples
    --------
    >>> import numpy as np
    >>> tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
    >>> save_waveform(tone, "tone.wav", dtype="int16")
    """
    sample_rate = sample_rate or DEFAULT_SAMPLE_RATE

    # Determine extension and normalize file path
    if ext:
        base, _ = os.path.splitext(file_path)
        file_path = f"{base}.{ext.lstrip('.')}"
    else:
        _, ext = os.path.splitext(file_path)
        ext = ext.lstrip(".").lower()

    # Create parent directories if they exist
    # os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Auto-select backend if not specified
    if backend is None:
        if ext == "wav":
            backend = "scipy"  # simple WAV support
        elif ext in {"flac", "ogg", "wav", "calf", "aiff", "aif"}:
            backend = "soundfile"  # more formats, better support
        else:
            backend = "soundfile"  # default fallback

    # Preprocess waveform: reshape, normalize, convert dtype
    waveform = preprocess_waveform(
        waveform,
        normalize=normalize,
        target_dtype=dtype,
        stereo_out=stereo_out,
    )

    # Detect number of channels for info/debug (optional)
    # channels = detect_channels(waveform)

    # Dispatch to backends
    try:
        if backend == "soundfile":
            import soundfile as sf  # type: ignore[reportMissingImports]

            sf.write(file_path, waveform, samplerate=sample_rate, **kwargs)
        elif backend == "scipy":
            if ext != "wav":
                raise ValueError("scipy backend supports only WAV files")
            from scipy.io.wavfile import write

            write(file_path, sample_rate, waveform)
        elif backend == "scitools":
            from scitools.sound import write  # type: ignore[reportMissingImports]

            write(data=waveform, filename=file_path, sample_rate=sample_rate)
        # elif backend == 'librosa':
        #     from scitools.sound import write
        #     write(data=waveform, filename=file_path, sample_rate=sample_rate)
        else:
            raise ValueError(f"Unsupported backend '{backend}'.")
    except Exception as e:
        logger.error(f"Failed to save waveform to '{file_path}': {e}")
        raise
    return file_path


def save_waveform_as_mp3(
    waveform: np.ndarray,
    file_path: str = "output.mp3",
    sample_rate: int | None = DEFAULT_SAMPLE_RATE,
    amplitude_int: int | None = DEFAULT_AMPLITUDE_INT,
    bitrate: str | None = DEFAULT_BITRATE,
    metadata: "dict[str, str] | None" = None,
):
    """
    Save waveform as an MP3 file using pydub and ffmpeg, with support for mono or stereo.

    Parameters
    ----------
    waveform : np.ndarray
        Waveform as float32 array, range [-1, 1].
    file_path : str, default="output.mp3"
        Output file path, must end with '.mp3'.
    sample_rate : int or None
        Sampling rate in Hz, defaults to 44100.
    amplitude_int : int or None
        Integer scale factor for converting float waveform to PCM int16.
        Defaults to 32767 (16-bit max).
    bitrate : str or None
        Bitrate string for MP3 encoding (e.g., '192k').
    metadata : dict or None
        Metadata tags to embed into the MP3 file.

    Returns
    -------
    str
        The file path of the saved audio.

    Examples
    --------
    >>> save_waveform_as_mp3(
    ...     tone, "Do4.mp3", metadata={"title": "Do4", "artist": "NoteGen"}
    ... )
    """
    # sample_rate = sample_rate or 44100
    # amplitude_int = amplitude_int or 32767  # Max 16-bit PCM amplitude
    # bitrate = bitrate or "192k"
    # tmp_wav = file_path.replace(".mp3", ".tmp.wav")
    # save_waveform(waveform, tmp_wav, samplerate=sample_rate)
    # audio = AudioSegment.from_wav(tmp_wav)
    # audio.export(file_path, format="mp3", bitrate=bitrate)
    try:
        from pydub import AudioSegment  # type: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError(
            "The 'pydub' package is required to save MP3 files. "
            "Install it with 'pip install pydub'."
        ) from e

    # Create parent directories if they exist
    # os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # Handle waveform shape and channel count
    if waveform.ndim == 1:
        channels = 1
        # Scale float waveform to int16
        int_waveform = np.int16(waveform * amplitude_int)
    elif waveform.ndim == 2 and waveform.shape[1] == 2:  # noqa: PLR2004
        channels = 2
        interleaved = (
            waveform.reshape(-1)
            if waveform.flags["C_CONTIGUOUS"]
            else np.ravel(waveform)
        )
        # Scale float waveform to int16
        int_waveform = np.int16(interleaved * amplitude_int)
    else:
        raise ValueError(
            "Waveform must be mono (1D) or stereo (2D with shape [samples, 2])"
        )
    try:
        audio = AudioSegment(
            int_waveform.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit PCM
            channels=channels,
        )
        # Export with metadata if provided
        audio.export(
            file_path,
            format="mp3",
            bitrate=bitrate,
            tags=metadata or {},
        )
    except Exception as e:
        logger.error(f"[ERROR] Failed to export MP3 to '{file_path}': {e}")
        raise RuntimeError(
            "Failed to save MP3. Ensure ffmpeg is installed and configured correctly."
        ) from e
