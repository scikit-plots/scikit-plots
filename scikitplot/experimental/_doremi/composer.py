# composer/composer.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=import-outside-toplevel
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=broad-exception-caught

"""
Generate audio tones from musical notes using sine wave synthesis with optional envelopes.

Supports:
- 🎼 Input from strings, lists, or structured dicts
- 🎶 Western and Solfège notation
- 🎵 Exporting individual notes to WAV/MP3
"""

import os

# import time  # ⏳
from typing import Union

import numpy as np
from pydantic import BaseModel, Field

from ... import logger
from ..._docstrings._docstring import interpd
from .config import DEFAULT_BITRATE, DEFAULT_SAMPLE_RATE
from .note import SHEET, _is_rest, sheet_to_note
from .note_io import save_waveform, save_waveform_as_mp3
from .synthesis import note_to_sine_wave

__all__ = [
    "compose_as_waveform",
    "export_notes_to_files",
]


# -------------------------------------------------------------------
# Pydantic Models for Structured Inputs
# -------------------------------------------------------------------


class NoteEntry(BaseModel):
    """NoteEntry."""

    note: str = Field(..., description="Note name (e.g., 'C', 'G#').")
    octave: int = Field(..., description="Octave number.")
    duration: float = Field(..., description="Duration in seconds.")


class CompositionSchema(BaseModel):
    """CompositionSchema."""

    notes: list[NoteEntry]


# -------------------------------------------------------------------
# Composition Generation
# -------------------------------------------------------------------


@interpd
def compose_as_waveform(
    composition: Union[str, list[tuple[str, int, float]], dict] = SHEET,
    envelope: "Union[str, callable[[np.ndarray, float], np.ndarray], None]" = "hann",
    **kwargs,
) -> np.ndarray:
    """
    Generate a concatenated waveform from a musical composition input.

    Parameters
    ----------
    composition : str or list of tuples or dict, optional
        Musical composition to synthesize. Supported formats:

        - str: Musical notation string, e.g. "C4-0.5 G4-0.5"
        - list of tuples: [('C', 4, 0.5), ('G', 4, 0.5), ...]
        - dict: {'notes': [{'note': 'C', 'octave': 4, 'duration': 0.5}, ...]}

        Defaults to an internal sample `SHEET`.
    envelope : str or callable, optional
        Envelope to shape amplitude over time.
        Choose from: 'hann', 'soft', 'triangular', or provide a custom function.
    **kwargs : dict
        Additional keyword arguments passed to `note_to_sine_wave`,
        such as amplitude, envelope, sample_rate, etc.

    Returns
    -------
    np.ndarray
        Concatenated audio waveform representing the full composition.

    Notes
    -----
    - Invalid or unrecognized note tokens are skipped silently.
    - Octave and duration must be numeric values.
    - Supports rests, generating silence of appropriate duration.
    """
    # if isinstance(sheet, dict):
    #     sheet = [(n.note, n.octave, n.duration) for n in CompositionSchema(**sheet).notes]
    composition = composition or SHEET  # Use default if None
    notes = sheet_to_note(composition)  # Normalize to list of (note, octave, duration)

    waveform_segments = []
    for note, octave, duration in notes:
        if _is_rest(note):
            # Append silence for rests
            silence = np.zeros(int(DEFAULT_SAMPLE_RATE * duration), dtype=np.float32)
            waveform_segments.append(silence)
        else:
            # Generate sine wave for note
            segment = note_to_sine_wave(
                note, octave, duration, envelope=envelope, **kwargs
            )
            waveform_segments.append(segment)

    result = (
        np.concatenate(waveform_segments)
        if waveform_segments
        else np.zeros(0, dtype=np.float32)
    )
    # logger.info(result.shape)
    return result  # noqa: RET504


# -------------------------------------------------------------------
# Export Utilities
# -------------------------------------------------------------------


def export_notes_to_files(
    notes: list[tuple[str, int, float]],
    output_dir: str = ".",
    file_format: str = "wav",
    filename_template: str = "{note}{octave}_{duration:.2f}s.{ext}",
    bitrate: str = DEFAULT_BITRATE,
    **kwargs,
):
    """
    Export each note in the list as an individual audio file.

    Parameters
    ----------
    notes : list of tuple
        List of notes to export, where each note is represented as (note, octave, duration).
    output_dir : str, optional
        Directory path to save the exported audio files. Defaults to current directory.
    file_format : str, optional
        Audio file format for export. Supported values: "wav", "mp3". Defaults to "wav".
    filename_template : str, optional
        Template string to format output filenames. Placeholders: note, octave, duration, ext.
    bitrate : str, optional
        Bitrate to use when exporting MP3 files. Defaults to module default bitrate.
    **kwargs : dict
        Additional keyword arguments forwarded to `note_to_sine_wave`
        (e.g., amplitude, envelope, sample_rate).

    Returns
    -------
    None
        Files are saved to disk; no return value.
    """
    os.makedirs(output_dir, exist_ok=True)

    for note, octave, duration in notes:
        waveform = note_to_sine_wave(note, octave, duration=duration, **kwargs)
        ext = file_format.lower()
        filename = filename_template.format(
            note=note,
            octave=octave,
            duration=duration,
            ext=ext,
        )
        full_path = os.path.join(output_dir, filename)

        if ext == "mp3":
            save_waveform_as_mp3(
                waveform,
                full_path,
                bitrate=bitrate,
                sample_rate=kwargs.get("sample_rate", DEFAULT_SAMPLE_RATE),
            )
        else:
            save_waveform(
                waveform,
                full_path,
                samplerate=kwargs.get("sample_rate", DEFAULT_SAMPLE_RATE),
            )


# -------------------------------------------------------------------
# CLI for Quick Testing
# -------------------------------------------------------------------

# %%
if __name__ == "__main__":
    import soundfile as sf  # type: ignore[reportMissingImports]

    music = compose_as_waveform(SHEET)
    sf.write("song.wav", music, samplerate=DEFAULT_SAMPLE_RATE)
    logger.info("song.wav saved.")
