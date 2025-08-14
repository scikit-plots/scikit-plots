# note.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# pylint: disable=broad-exception-caught

"""
🎼 Generate/Handle Composition SHEET.

This module supports parsing, validating, and converting musical sheet notation
from Western and Solfège formats. Notes can be represented in strings like:
    G4-1  -> Note G, 4th octave, quarter duration
    La#5-0.5 -> La#, 5th octave, eighth note

- Note normalization (sharps, Unicode symbols, Solfège)
- Frequency calculation with A4 tuning reference

Examples
--------
>>> parse_note_token("G4-1")
SheetNoteToken(note='G', octave=4, duration=1.0)

>>> sheet_to_note("Do4-1 Re4-0.5 Mi4-1")
[('Do', 4, 1.0), ('Re', 4, 0.5), ('Mi', 4, 1.0)]

Notes
-----
- https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fpages.mtu.edu%2F~suits%2FScaleFreqs.xls

+-------+---------+-----------+
| notes | notesDo | frequency |
+=======+=========+===========+
| C0	| Do0	  | 16,35     |
+-------+---------+-----------+
| C4	| Do4	  | 261,63    |
+-------+---------+-----------+
| C#4	| Do#4	  | 277,18    |
+-------+---------+-----------+
| D4	| Re4	  | 293,66    |
+-------+---------+-----------+
| D#4	| Re#4	  | 311,13    |
+-------+---------+-----------+
| E4	| Mi4	  | 329,63    |
+-------+---------+-----------+
| F4	| Fa4	  | 349,23    |
+-------+---------+-----------+
| F#4	| Fa#4	  | 369,99    |
+-------+---------+-----------+
| G4	| Sol4	  | 392,00    |
+-------+---------+-----------+
| G#4	| Sol#4	  | 415,30    |
+-------+---------+-----------+
| A4	| La4	  | 440,00    |
+-------+---------+-----------+
| A#4	| La#4	  | 466,16    |
+-------+---------+-----------+
| B4	| Si4	  | 493,88    |
+-------+---------+-----------+
| B8	| Si8	  | 7902,13   |
+-------+---------+-----------+
"""

import json
import re
from dataclasses import dataclass
from typing import Literal, Optional, Union

import pandas as pd

from ... import logger
from ..._docstrings._docstring import interpd

__all__ = [
    "SHEET",
    "export_sheet",
    "serialize_sheet",
    "sheet_add_frequency",
    "sheet_converter",
    "sheet_to_note",
]

# ----------------------------------------------------------------------------
# 🎼 Constants for Musical Notes (Western and Solfège)
# ----------------------------------------------------------------------------
# Supported note tokens in Solfège notation
NOTE_TOKENS_SOLFEGE = [
    "Do",
    "Do#",
    "Re",
    "Re#",
    "Mi",
    "Fa",
    "Fa#",
    "Sol",
    "Sol#",
    "La",
    "La#",
    "Si",
]

# Supported note tokens in Western notation
NOTE_TOKENS_WESTERN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Recognized rest tokens
NOTE_TOKENS_REST = {"r", "rest"}

# Mapping from flat to sharp notes
NOTE_TOKENS_FLAT_TO_SHARP = {
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "Reb": "Do#",
    "Mib": "Re#",
    "Solb": "Fa#",
    "Lab": "Sol#",
    "Sib": "La#",
}

# Mappings between Western and Solfège
NOTE_TOKENS_SOLFEGE_TO_WESTERN = dict(zip(NOTE_TOKENS_SOLFEGE, NOTE_TOKENS_WESTERN))
NOTE_TOKENS_WESTERN_TO_SOLFEGE = dict(zip(NOTE_TOKENS_WESTERN, NOTE_TOKENS_SOLFEGE))

# Note to index mapping (used for frequency calculations)
NOTE_INDEX_REFERENCE = {
    **{n: i for i, n in enumerate(NOTE_TOKENS_SOLFEGE)},
    **{n: i for i, n in enumerate(NOTE_TOKENS_WESTERN)},
}

# Regex to match valid note tokens in the form: NoteOctave-Duration (e.g., G4-1.0)
NOTE_PATTERN = re.compile(
    r"^([A-Ga-g]|Do|Re|Mi|Fa|Sol|La|Si)([#b♭♯]?)(\d)-(\d*\.?\d+)$"
)
SHEET_NOTE_PATTERN = NOTE_PATTERN

# ----------------------------------------------------------------------------
# 🎼 Sheet Note Representation
# ----------------------------------------------------------------------------


@dataclass
class SheetNoteToken:
    """Dataclass representing a parsed musical note."""

    note: str
    octave: int
    duration: float
    # Octave number (4 = middle octave).
    a4_freq: float = 440.0  # Reference Frequency in Hertz for A4
    a4_midi: int = 69  # MIDI note number for A4

    def to_str(self) -> str:
        """Return note as string."""
        return f"{self.note}{self.octave}-{self.duration}"

    def to_dict(self) -> dict:
        """Return note as dictionary."""
        return {"note": self.note, "octave": self.octave, "duration": self.duration}

    def to_list(self) -> list:
        """Return note as list."""
        return [self.note, self.octave, self.duration]

    def is_solfege(self) -> bool:
        """Check if note is Solfège."""
        return self.note in NOTE_TOKENS_SOLFEGE_TO_WESTERN

    def to_solfege(self) -> str:
        """Convert note to Solfège."""
        return NOTE_TOKENS_WESTERN_TO_SOLFEGE.get(self.note, self.note)

    def is_western(self) -> bool:
        """Check if note is Western."""
        return self.note in NOTE_TOKENS_WESTERN_TO_SOLFEGE

    def to_western(self) -> str:
        """Convert note to Western."""
        return NOTE_TOKENS_SOLFEGE_TO_WESTERN.get(self.note, self.note)

    def frequency(self) -> float:
        """Calculate frequency using MIDI formula."""
        midi_number = (self.octave + 1) * 12 + NOTE_INDEX_REFERENCE[self.note]
        return 440.0 * (2 ** ((midi_number - 69) / 12))

    def frequency2(self) -> float:
        """Alternative method to compute frequency from A4."""
        semitone_offset = 12 * (self.octave - 4) + (
            NOTE_INDEX_REFERENCE[self.note] - NOTE_INDEX_REFERENCE["A"]
        )
        return self.a4_freq * (2 ** (semitone_offset / 12))

    def __str__(self):
        return self.to_str()


# ----------------------------------------------------------------------------
# 🪰 Utility Functions
# ----------------------------------------------------------------------------


def _is_rest(note: str) -> bool:
    """Check if a note represents a rest."""
    return note.lower() in NOTE_TOKENS_REST


def normalize_note(note: str) -> str:
    """Normalize note by converting flats/sharps and validating it."""
    note = note.strip().capitalize().replace("♯", "#").replace("♭", "b")
    note = NOTE_TOKENS_FLAT_TO_SHARP.get(note, note)
    if note not in NOTE_INDEX_REFERENCE:
        raise ValueError(f"Unsupported note '{note}'")
    return note


def parse_note_token(token: str) -> Optional[SheetNoteToken]:
    """Parse a token like 'G4-1' or 'La#5-0.5' into a SheetNoteToken."""
    match = SHEET_NOTE_PATTERN.match(token)
    if not match:
        return None
    name, accidental, octave, duration = match.groups()
    try:
        note = normalize_note(name + accidental)
        return SheetNoteToken(note, int(octave), float(duration))
    except Exception as e:
        logger.warning(f"Failed to parse token '{token}': {e}")
        return None


def sheet_cleaner(sheet: str) -> str:
    """Remove comments and extra spacing from a sheet string."""
    return " ".join(
        line.split("#")[0].strip()
        for line in sheet.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ).replace(" - ", " ")


def sheet_parser(sheet: str) -> list[SheetNoteToken]:
    """Parse cleaned sheet into list of SheetNoteToken objects."""
    cleaned = sheet_cleaner(sheet)
    tokens = cleaned.split()
    return [n for t in tokens if (n := parse_note_token(t)) is not None]


def sheet_to_note(sheet: Union[str, list, dict]) -> list[tuple[str, int, float]]:
    """Convert input sheet (str/list/dict) to a list of (note, octave, duration)."""
    notes = []
    if isinstance(sheet, str):
        return [(n.note, n.octave, n.duration) for n in sheet_parser(sheet)]
    if isinstance(sheet, list):
        for item in sheet:
            try:
                n, o, d = item
                notes.append((normalize_note(n), int(o), float(d)))
            except Exception as e:
                logger.warning(f"Invalid list entry {item}: {e}")
    elif isinstance(sheet, dict):
        for item in sheet.get("notes", []):
            try:
                notes.append(
                    (
                        normalize_note(item["note"]),
                        int(item["octave"]),
                        float(item["duration"]),
                    )
                )
            except Exception as e:
                logger.warning(f"Invalid dict entry {item}: {e}")
    return notes


def sheet_add_frequency(
    sheet: Union[str, list, dict, None] = None,
) -> list[tuple[str, int, float, float]]:
    """
    Convert input sheet (str, list, or dict) to a list of tuples with note frequency.

    Parameters
    ----------
    sheet : str or list or dict or None
        Musical input in one of the supported formats:
        - str  : Sheet string like "C4-1 D4-0.5"
        - list : List of (note, octave, duration)
        - dict : Dictionary with "notes" key as list of note dicts

    Returns
    -------
    list of tuple
        List of (note, octave, duration, frequency) for each parsed note.

    Examples
    --------
    >>> sheet_add_frequency("A4-1")
    [('A', 4, 1.0, 440.0)]
    """
    sheet = sheet or SHEET  # Use internal fallback
    sheet = sheet_to_note(sheet)  # Convert to (note, octave, duration)

    result = []
    for n, o, d in sheet:
        try:
            norm_note = normalize_note(n)
            token = SheetNoteToken(norm_note, o, d)
            freq = round(token.frequency(), 2)  # Optional rounding
            result.append((norm_note, o, d, freq))
        except Exception as e:
            logger.warning(f"[sheet_add_frequency] Skipping {n}{o}-{d}: {e}")

    return result


def sheet_to_dataframe(sheet) -> pd.DataFrame:
    """Integrate with pandas to allow inspection/export of notes and frequencies."""
    data = sheet_add_frequency(sheet)
    return pd.DataFrame(data, columns=["note", "octave", "duration", "frequency"])


def sheet_converter(
    sheet: Union[str, list, dict, None] = None,
    add_frequency: bool = True,
    return_mode: Literal["str", "list", "dict", "df"] = "dict",
) -> Union[str, list, dict, pd.DataFrame]:
    """
    Display parsed notes or note frequencies from a musical sheet.

    Parameters
    ----------
    sheet : str or list or dict or None
        Musical input in one of the supported formats:
        - str  : Sheet string like "C4-1 D4-0.5"
        - list : List of (note, octave, duration)
        - dict : Dictionary with "notes" key as list of note dicts
        If None, uses internal default `SHEET`.

    add_frequency : bool, default=True
        If True, include frequency in output.

    return_mode : {'str', 'list', 'dict', 'df'}, default='dict'
        Output format:
        - 'str'  : Multiline string
        - 'list' : List of strings
        - 'dict' : List of dicts
        - 'df'   : pandas DataFrame

    Returns
    -------
    str or list or dict or pandas.DataFrame
        Formatted note data.

    Examples
    --------
    >>> sheet_converter(return_mode='df')
    ...     note octave duration frequency
    ... 0	G	 4	    0.50	 392.00
    """
    sheet = sheet or SHEET  # Use internal fallback
    sheet = sheet_to_note(sheet)  # Convert to (note, octave, duration)

    result = []
    for n, o, d in sheet:
        try:
            norm_note = normalize_note(n)
            note = SheetNoteToken(norm_note, o, d)
            note_dict = note.to_dict()
            if add_frequency:
                freq = round(note.frequency(), 2)  # Optional rounding
                note_dict["frequency"] = freq

            if return_mode in {"dict", "df"}:
                result.append(note_dict)
            elif return_mode == "list":
                if add_frequency:
                    result.append((*note.to_list(), freq))
                else:
                    result.append(note.to_list())
            elif add_frequency:
                result.append(f"{note.to_str()} beat -> {freq:.2f} Hz")
            else:
                result.append(str(note))
        except Exception as e:
            logger.warning(f"[sheet_add_frequency] Skipping {n}{o}-{d}: {e}")
    if return_mode == "str":
        return "\n".join(result)
    if return_mode == "df":
        return pd.DataFrame(result)
    return result


def serialize_sheet(
    sheet: Union[str, list, dict, None] = None,
    save_format: Literal["json", "yaml"] = "json",
    add_frequency: bool = True,
) -> str:
    """
    Serialize sheet notes to JSON or YAML string.

    Parameters
    ----------
    sheet : str or list or dict or None
        Musical sheet input (same supported formats as sheet_converter).

    save_format : {'json', 'yaml'}, default='json'
        Serialization save_format.

    add_frequency : bool, default=True
        Include frequency in output.

    Returns
    -------
    str
        Serialized string in requested save_format.

    Examples
    --------
    >>> print(serialize_sheet("A4-1", save_format="json"))
    >>> print(serialize_sheet("A4-1", save_format="yaml"))
    """
    # Get list of dicts with frequency
    notes_data = sheet_converter(sheet, add_frequency=add_frequency, return_mode="dict")
    if save_format == "json":
        return json.dumps(notes_data, indent=2)
    if save_format == "yaml":
        import yaml  # pip install pyyaml

        return yaml.dump(notes_data, sort_keys=False)
    raise ValueError(
        f"Unsupported save_format: {save_format}. Choose 'json' or 'yaml'."
    )


def export_sheet(sheet, path, fmt="json"):
    """Enable serialization of compositions or note sheets."""
    data = sheet_add_frequency(sheet)
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    elif fmt == "yaml":
        import yaml  # pip install pyyaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f)


# ----------------------------------------------------------------------------
# 🎼 Sample Sheet (Happy Birthday)
# ----------------------------------------------------------------------------

SHEET = """
# Format: NoteOctave-Duration
# NoteOctave: Musical note + octave number (e.g., G4 means G in the 4th octave)
# Duration: Length of the note (relative)
#   1   = quarter note
#  0.5  = eighth note
#   2   = half note
#
# Happy Birthday Melody — Western notation with lyrics:

G4-0.5    -  G4-0.25   -  A4-0.5    -  G4-0.5    -  C5-0.5    -  B4-1
# "Happy"    "birth-"   "day"     "to"     "you"

G4-0.5    -  G4-0.25   -  A4-0.5    -  G4-0.5    -  D5-0.5    -  C5-1
# "Happy"    "birth-"   "day"     "to"     "you"

G4-0.5    -  G4-0.25   -  G5-0.5    -  E5-0.5    -  C5-0.5    -  B4-0.5    -  A4-1
# "Happy"    "birth-"   "day"     "dear"    "[Name]"

F5-0.5    -  F5-0.25   -  E5-0.5    -  C5-0.5    -  D5-0.5    -  C5-1
# "Happy"    "birth-"   "day"     "to"     "you"
"""

interpd.register(_SHEET=SHEET)
