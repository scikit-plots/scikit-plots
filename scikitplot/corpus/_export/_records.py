# scikitplot/corpus/_export/_records.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
The record contract of the text exports (JSON, JSON Lines, YAML, TOML).

A record is :meth:`CorpusDocument.to_dict` output, plus ``embedding`` and
``embedding_dtype`` when an embedding is written. This module states, field
by field, what a record may hold, and checks it on both sides:

* on export, :func:`export_problem` finds the first value a format cannot
  hold, so the error names the document and the field;
* on load, :func:`record_problem` refuses anything outside the contract
  (a wrong type, an unknown key, a YAML or TOML date, an embedding value
  that overflows its dtype), so a malformed or hostile file never loads as
  plausible documents.

Notes
-----
**Developer note.** :data:`FIELD_KINDS` must list exactly the keys of
:meth:`CorpusDocument.to_dict`; a test holds it equal to the dataclass
annotations, so a new field cannot be exported unchecked.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__: list[str] = []

#: Record key holding the embedding's NumPy dtype name.
EMBEDDING_DTYPE_KEY = "embedding_dtype"

#: Embedding dtypes a record may name. An embedding array of another dtype
#: (integer, bool, complex, extended precision) is refused on export: its
#: values would change, or not load back.
EMBEDDING_DTYPES: tuple[str, ...] = ("float16", "float32", "float64")

#: Field name to kind. ``?`` marks a field that may be ``None`` or absent.
FIELD_KINDS: Mapping[str, str] = {
    "schema_version": "str",
    "doc_id": "str",
    "input_path": "str",
    "chunk_index": "int",
    "text": "str?",
    "section_type": "str",
    "chunking_strategy": "str",
    "language": "str?",
    "char_start": "int?",
    "char_end": "int?",
    "metadata": "mapping",
    "source_type": "str",
    "source_title": "str?",
    "source_author": "str?",
    "source_date": "str?",
    "collection_id": "str?",
    "url": "str?",
    "doi": "str?",
    "isbn": "str?",
    "page_number": "int?",
    "paragraph_index": "int?",
    "line_number": "int?",
    "parent_doc_id": "str?",
    "embedding_manifest_id": "str?",
    "act": "int?",
    "scene_number": "int?",
    "timecode_start": "float?",
    "timecode_end": "float?",
    "confidence": "float?",
    "ocr_engine": "str?",
    "bbox": "bbox?",
    "normalized_text": "str?",
    "raw_text": "str?",
    "tokens": "strlist?",
    "lemmas": "strlist?",
    "stems": "strlist?",
    "keywords": "strlist?",
    "script": "str?",
    "script_direction": "str?",
    "grapheme_count": "int?",
    "codepoint_count": "int?",
    "is_mixed_script": "bool?",
    "script_spans": "list?",
    "chunking_unit": "str?",
    "semanteme_count": "int?",
    "morphemes": "strlist?",
    "determinative_groups": "list?",
    "script_model_version": "str?",
    "modality": "str?",
    "frame_index": "int?",
    "raw_shape": "intlist?",
    "raw_dtype": "str?",
    "content_hash": "str?",
    "embedding": "embedding?",
    EMBEDDING_DTYPE_KEY: "str?",
}

#: Keys a record must hold.
REQUIRED_KEYS: frozenset = frozenset(
    k for k, kind in FIELD_KINDS.items() if not kind.endswith("?")
)

_BBOX_LENGTH = 4
_SURROGATES = range(0xD800, 0xE000)


# --------------------------------------------------------------------------
# Export side
# --------------------------------------------------------------------------


def embedding_dtype(embedding: Any) -> str:
    """
    Return the dtype name recorded for an embedding, or raise.

    Parameters
    ----------
    embedding : array-like
        The document's embedding.

    Returns
    -------
    str
        ``float16``, ``float32`` or ``float64``. A Python list of real
        numbers has no dtype: it is recorded as ``float64``, the type of a
        JSON number.

    Raises
    ------
    ValueError
        For an embedding that is not one-dimensional, or an array whose
        dtype is not one of :data:`EMBEDDING_DTYPES`.
    """
    if isinstance(embedding, np.ndarray):
        if embedding.ndim != 1:
            raise ValueError(
                f"the embedding has shape {embedding.shape}; text exports hold one vector"
            )
        if embedding.dtype.name not in EMBEDDING_DTYPES:
            raise ValueError(
                f"the embedding dtype {embedding.dtype.name} is not one of {EMBEDDING_DTYPES}; "
                "cast it (for example .astype(np.float32)) or export as pickle"
            )
        return embedding.dtype.name
    values = list(embedding)
    if not all(
        isinstance(v, numbers.Real) and not isinstance(v, (bool, np.bool_))
        for v in values
    ):
        raise ValueError("the embedding is not a flat sequence of real numbers")
    return "float64"


def _scalar_problem(value: Any, fmt: str, path: str) -> str | None:
    """Return why ``fmt`` cannot hold a scalar ``value``, or ``None``."""
    json_like = fmt in ("json", "jsonl")
    if isinstance(value, float) and json_like and math.isinf(value):
        return f"{path} is {value} (JSON has no infinity)"
    if (
        isinstance(value, float)
        and json_like
        and math.isnan(value)
        and not path.startswith(("$.metadata", "$.embedding"))
    ):
        return (
            f"{path} is nan (JSON has no NaN; only metadata and embeddings may hold it)"
        )
    if (
        isinstance(value, str)
        and fmt != "yaml"
        and any(ord(c) in _SURROGATES for c in value)
    ):
        return f"{path} holds a lone surrogate, which {fmt} cannot encode"
    return None


def export_problem(tree: Any, fmt: str, path: str = "$") -> str | None:
    """
    Return the first value ``fmt`` cannot hold, as a path and a reason.

    Parameters
    ----------
    tree : object
        A record in the serialisation data model.
    fmt : {"json", "jsonl", "yaml", "toml"}
        Target format.
    path : str, default="$"
        Path of ``tree``.

    Returns
    -------
    str or None
        ``None`` when ``tree`` can be written and read back unchanged
        (apart from the documented ``NaN`` -> ``null`` in JSON metadata).
    """
    if isinstance(tree, dict):
        children = [
            (f"{path}.{key}", value, path != "$") for key, value in tree.items()
        ]
    elif isinstance(tree, list):
        children = [
            (f"{path}[{index}]", value, True) for index, value in enumerate(tree)
        ]
    else:
        return _scalar_problem(tree, fmt, path)
    for child_path, value, null_dropped in children:
        if fmt == "toml" and value is None and null_dropped:
            return f"{child_path} is None (TOML has no null and would drop it)"
        found = export_problem(value, fmt, child_path)
        if found:
            return found
    return None


# --------------------------------------------------------------------------
# Load side
# --------------------------------------------------------------------------


def _model_problem(value: Any, path: str) -> str | None:
    """Return where ``value`` leaves the JSON data model, or ``None``."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return None
    if isinstance(value, list):
        children = [(f"{path}[{index}]", item) for index, item in enumerate(value)]
    elif isinstance(value, dict):
        bad_keys = [key for key in value if not isinstance(key, str)]
        if bad_keys:
            return f"{path} has a {type(bad_keys[0]).__name__} key"
        children = [(f"{path}.{key}", item) for key, item in value.items()]
    else:
        return f"{path} is a {type(value).__name__} (an unquoted YAML/TOML date or time is not a string)"
    return next(
        (found for found in (_model_problem(item, p) for p, item in children) if found),
        None,
    )


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _kind_problem(key: str, kind: str, value: Any) -> str | None:
    """Return why ``value`` is not of ``kind``, or ``None``."""
    if kind.endswith("?"):
        if value is None:
            return None
        kind = kind[:-1]
    checks = {
        "str": lambda v: isinstance(v, str),
        "int": _is_int,
        "float": _is_real,
        "bool": lambda v: isinstance(v, bool),
        "mapping": lambda v: isinstance(v, dict),
        "list": lambda v: isinstance(v, list),
        "strlist": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
        "intlist": lambda v: isinstance(v, list) and all(_is_int(x) for x in v),
        "bbox": lambda v: (
            isinstance(v, list)
            and len(v) == _BBOX_LENGTH
            and all(_is_real(x) for x in v)
        ),
        "embedding": lambda v: (
            isinstance(v, list) and all(x is None or _is_real(x) for x in v)
        ),
    }
    if checks[kind](value):
        return None
    expected = {
        "str": "a string",
        "int": "an integer",
        "float": "a number",
        "bool": "true or false",
        "mapping": "a mapping",
        "list": "a list",
        "strlist": "a list of strings",
        "intlist": "a list of integers",
        "bbox": "a list of 4 numbers",
        "embedding": "a list of numbers",
    }[kind]
    return f"{key} is {type(value).__name__}, expected {expected}"


def record_problem(record: Any) -> str | None:
    """
    Return why ``record`` breaks the contract, or ``None``.

    Parameters
    ----------
    record : object
        One parsed record.

    Returns
    -------
    str or None

    Notes
    -----
    An ``embedding`` without ``embedding_dtype`` (a file from an earlier
    exporter) is accepted and rebuilt as ``float64``, the type of a JSON
    number; ``embedding_dtype`` without ``embedding`` is refused.
    """
    if not isinstance(record, dict):
        return f"it is {type(record).__name__}, expected a mapping"
    unknown = sorted(str(k) for k in record if k not in FIELD_KINDS)
    missing = sorted(REQUIRED_KEYS - set(record))
    dtype = record.get(EMBEDDING_DTYPE_KEY)
    checks = [
        f"unknown keys {unknown[:5]}" if unknown else None,
        f"missing required keys {missing}" if missing else None,
        _model_problem(record, "$"),
        *(
            _kind_problem(key, FIELD_KINDS.get(key, "str?"), value)
            for key, value in record.items()
        ),
        (
            f"{EMBEDDING_DTYPE_KEY} without an embedding"
            if dtype is not None and record.get("embedding") is None
            else None
        ),
        (
            f"{EMBEDDING_DTYPE_KEY} {dtype!r} is not one of {EMBEDDING_DTYPES}"
            if dtype is not None and dtype not in EMBEDDING_DTYPES
            else None
        ),
    ]
    return next((problem for problem in checks if problem), None)


def embedding_array(values: list, dtype: str) -> np.ndarray:
    """
    Rebuild an embedding exactly, ``None`` (JSON's ``NaN``) included.

    Raises
    ------
    ValueError
        If a finite value overflows ``dtype``.
    """
    numbers_ = [math.nan if v is None else v for v in values]
    with np.errstate(over="ignore"):
        try:
            array = np.asarray(numbers_, dtype=np.dtype(dtype))
        except OverflowError:
            raise ValueError(f"an embedding value overflows {dtype}") from None
    source = np.asarray(
        [float(v) if _fits_float(v) else math.inf for v in numbers_], dtype=np.float64
    )
    if np.any(np.isfinite(source) & ~np.isfinite(array.astype(np.float64))):
        raise ValueError(f"an embedding value overflows {dtype}")
    return array


def _fits_float(value: Any) -> bool:
    try:
        float(value)
    except OverflowError:
        return False
    return True
