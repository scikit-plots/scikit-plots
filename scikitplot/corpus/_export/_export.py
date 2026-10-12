# scikitplot/corpus/_export/_export.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._export
=========================

Multi-format corpus export for
:class:`~scikitplot.corpus._schema.CorpusDocument` lists.

All formats are written **atomically**: data is first written to a
``*.tmp`` sibling file, then renamed over the target path. This prevents
partially-written files from being read by downstream processes if the
export is interrupted.

Supported formats
-----------------
All values of :class:`~scikitplot.corpus._schema.ExportFormat`:

:attr:`ExportFormat.CSV`
    Flat CSV via stdlib ``csv.DictWriter``. Zero optional dependencies.
    Metadata dict keys are flattened into top-level columns via
    :meth:`~scikitplot.corpus._schema.CorpusDocument.to_flat_dict`.

:attr:`ExportFormat.JSONL`
    One JSON object per line (JSON Lines). Zero optional dependencies.
    Metadata dict is preserved as a nested object.

:attr:`ExportFormat.JSON`
    JSON array of document objects. Zero optional dependencies. Indented
    by default for readability.

:attr:`ExportFormat.YAML`
    YAML sequence of document mappings. Requires ``pip install pyyaml``.

:attr:`ExportFormat.TOML`
    TOML document whose ``documents`` key holds an array of tables (TOML
    has no top-level array). Requires ``pip install tomli-w``; reading uses
    :mod:`tomllib` (Python 3.11+) or ``tomli``.

The four text formats go through :mod:`scikitplot.corpus._serialize`, the
same contract as evaluation reports: strict JSON (``NaN`` written as
``null``, an infinity refused), line-break characters escaped, YAML read
without aliases. A document with an embedding also records
``embedding_dtype``, so :func:`load_documents` restores the vector exactly.

:attr:`ExportFormat.PICKLE`
    Python ``pickle`` of the full ``list[CorpusDocument]``. Preserves
    embeddings and all metadata exactly. Zero optional dependencies.

:attr:`ExportFormat.JOBLIB`
    ``joblib.dump`` of the document list. Faster than pickle for large
    NumPy embedding arrays. Requires ``pip install joblib``.

:attr:`ExportFormat.NUMPY`
    ``np.save`` of the embedding matrix only, shape ``(n, dim)``.
    Raises ``ValueError`` if no documents carry embeddings.
    Requires ``numpy`` (already a hard dependency of this package).

:attr:`ExportFormat.PANDAS`
    ``pandas.DataFrame.to_csv`` — identical column layout to
    :attr:`CSV` but goes through pandas, enabling richer dtype handling.
    Requires ``pip install pandas``.

:attr:`ExportFormat.PARQUET`
    ``pandas.DataFrame.to_parquet``. Columnar, compressed, fast.
    Requires ``pip install pandas pyarrow`` (or ``fastparquet``).

:attr:`ExportFormat.POLARS`
    ``polars.DataFrame.write_parquet``. Requires ``pip install polars``.

:attr:`ExportFormat.HUGGINGFACE`
    ``datasets.Dataset.save_to_disk``. Produces a HuggingFace Arrow
    dataset directory tree rooted at ``output_path``.
    Requires ``pip install datasets``.

:attr:`ExportFormat.MLFLOW`
    Logs the documents as an MLflow artifact. ``output_path`` is used
    as the ``artifact_path`` prefix inside the active run.
    Requires ``pip install mlflow`` and an active MLflow run
    (``mlflow.start_run()``).

Loading back
------------
:func:`load_documents` reads JSON, JSON Lines, YAML and TOML exports
(safe: no code runs; every record is checked field by field, see
:mod:`._records`) and, with ``trusted=True``, pickle and joblib. Tabular
formats (CSV, pandas, Parquet, polars) and NumPy cannot rebuild every field
and HuggingFace is a dataset directory, so loading them is refused rather
than returned empty. An MLflow export is a JSON Lines artifact of a run:
download ``corpus_documents.jsonl`` and load that file.

Python compatibility
--------------------
Python 3.9-3.15. Only ``csv``, ``json``, and ``pickle`` (all stdlib)
and ``numpy`` are hard requirements. All other backends are optional.
"""  # noqa: D205, D400

from __future__ import annotations

import csv
import hashlib
import io
import logging
import pathlib
import pickle
from collections.abc import Mapping
from typing import Any, Dict, List, Optional  # noqa: F401

import numpy as np

from .. import _serialize
from .._atomic import atomic_write_bytes, atomic_write_path
from .._schema import CorpusDocument, ExportFormat
from ._records import (
    EMBEDDING_DTYPE_KEY,
    EMBEDDING_DTYPES,  # noqa: F401 - re-exported for callers of this module
    embedding_array,
    embedding_dtype,
    export_problem,
    record_problem,
)

logger = logging.getLogger(__name__)

__all__ = [
    "export_documents",
    "load_documents",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: JSON indent for :attr:`ExportFormat.JSON` output.
_JSON_INDENT: int = 2

#: CSV encoding used by all text-based exporters.
_CSV_ENCODING: str = "utf-8"

#: Export formats written and read through :mod:`.._serialize`.
TEXT_FORMATS: dict[ExportFormat, str] = {
    ExportFormat.JSON: "json",
    ExportFormat.JSONL: "jsonl",
    ExportFormat.YAML: "yaml",
    ExportFormat.TOML: "toml",
}

#: Top-level key of a TOML export (TOML has no top-level array).
TOML_DOCUMENTS_KEY = "documents"

#: File suffix to format, for :func:`load_documents` without ``format``.
LOAD_SUFFIXES: dict[str, ExportFormat] = {
    ".json": ExportFormat.JSON,
    ".jsonl": ExportFormat.JSONL,
    ".ndjson": ExportFormat.JSONL,
    ".yaml": ExportFormat.YAML,
    ".yml": ExportFormat.YAML,
    ".toml": ExportFormat.TOML,
    ".pkl": ExportFormat.PICKLE,
    ".pickle": ExportFormat.PICKLE,
    ".joblib": ExportFormat.JOBLIB,
    # Recognised so that loading them is refused with the reason, not "unknown suffix".
    ".csv": ExportFormat.CSV,
    ".parquet": ExportFormat.PARQUET,
    ".npy": ExportFormat.NUMPY,
}

#: Stable, identity-first column ordering for CSV/pandas exports.
#:
#: Used both by :func:`_compute_csv_fieldnames` (as the priority prefix) and
#: by the empty-document fast-path in :func:`_export_csv` /
#: :func:`_export_pandas` to write a header-only file instead of zero bytes.
#: A header-only file lets ``pd.read_csv()`` return an empty DataFrame with
#: named columns rather than raising ``pandas.errors.EmptyDataError``.
#:
#: **Rationale for module-level constant:**  the list was previously
#: inlined inside ``_compute_csv_fieldnames``, which meant it was only
#: reachable at runtime when documents existed.  Hoisting it here makes the
#: canonical column order the single source of truth for all callers.
_CSV_IDENTITY_ORDER: list[str] = [
    "doc_id",
    "input_path",
    "chunk_index",
    # Three-tier OCR text fields — grouped for easy visual comparison:
    "text",  # chunked form
    "raw_text",  # pre-normalization OCR output (None for non-OCR sources)
    "normalized_text",  # post-TextNormalizer NFKC form
    "section_type",
    "chunking_strategy",
    "language",
    "source_type",
    "source_title",
    "source_author",
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def export_documents(
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    format: ExportFormat,
    *,
    include_embedding: bool = True,
    json_indent: int | None = _JSON_INDENT,
    parquet_compression: str = "snappy",
) -> pathlib.Path:
    """
    Export a list of documents to ``output_path`` in the given format.

    Parameters
    ----------
    documents : list of CorpusDocument
        Documents to export. May be empty (produces an empty file/dataset).
    output_path : pathlib.Path
        Destination file or directory path.

        - File formats (CSV, JSON, JSONL, Pickle, Joblib, NumPy, pandas,
          Parquet, Polars): path to the output file.
        - Directory formats (HuggingFace, MLflow): path to the root
          directory / artifact path.
    format : ExportFormat or str
        Target export format, or its string value (``"jsonl"``).
    include_embedding : bool, optional
        When ``True`` (default), embedding vectors are included in the
        output where the format supports them (JSONL, JSON, Pickle,
        Joblib, NumPy). Embeddings are always included for NumPy.
        For CSV and Parquet (tabular), embeddings are excluded regardless
        of this flag to avoid storing variable-length arrays in cells.
    json_indent : int or None, optional
        Indentation for JSON output. ``None`` produces compact JSON.
        Default: ``2``.
    parquet_compression : str, optional
        Compression codec for Parquet output (``"snappy"``, ``"gzip"``,
        ``"brotli"``, ``"zstd"``, ``"none"``). Default: ``"snappy"``.

    Returns
    -------
    pathlib.Path
        The path that was written to (same as ``output_path``).

    Raises
    ------
    ValueError
        If ``format`` is :attr:`ExportFormat.NUMPY` and no documents have
        embeddings, or if the embedding dimensions are inconsistent; for a
        text format, if a value cannot be represented (an infinity in JSON,
        ``None`` in a TOML array or in metadata for TOML). The message names
        the document.
    TypeError
        For a text format, if a metadata value is not serialisable (a
        ``set``, a ``datetime``, a custom object); the message names the
        document.
    ImportError
        If the required optional library for the format is not installed.
    OSError
        If the output directory cannot be created or the file cannot be
        written.

    See Also
    --------
    scikitplot.corpus._schema.ExportFormat : Enumeration of all formats.

    Notes
    -----
    **Atomic writes:** All file-based formats are written to a ``.tmp``
    sibling first, then renamed atomically. Interrupted exports leave no
    partial files at the final path.

    **Embedding in tabular formats:** CSV and Parquet omit embeddings
    because storing a float32 vector per row in a tabular cell is
    impractical. Use JSON, JSON Lines, YAML or TOML (safe to load back) or
    PICKLE/JOBLIB to preserve embeddings.

    **Text formats:** metadata must hold JSON-like values (``str``,
    numbers, ``bool``, ``None``, lists, string-keyed mappings; NumPy scalars
    are converted). Tuples come back as lists. JSON has no ``NaN``: a ``NaN``
    in metadata is written as ``null`` and reads back as ``None`` (YAML and
    TOML keep ``NaN``; embeddings restore it in every format).

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from scikitplot.corpus import CorpusDocument, ExportFormat
    >>> docs = [
    ...     CorpusDocument.create(
    ...         "notes.txt", i, f"Paragraph {i}.", embedding=np.full(3, i, np.float32)
    ...     )
    ...     for i in range(2)
    ... ]
    >>> out = Path(tempfile.mkdtemp())

    CSV (zero dependencies; no embeddings):

    >>> export_documents(docs, out / "corpus.csv", ExportFormat.CSV).name
    'corpus.csv'

    JSON Lines with embeddings, loadable back exactly:

    >>> path = export_documents(docs, out / "corpus.jsonl", ExportFormat.JSONL)
    >>> load_documents(path)[1].embedding
    array([1., 1., 1.], dtype=float32)

    NumPy embedding matrix:

    >>> export_documents(docs, out / "embeddings.npy", ExportFormat.NUMPY).name
    'embeddings.npy'
    """
    output_path = pathlib.Path(output_path)
    if not isinstance(format, ExportFormat) and isinstance(format, str):
        format = _as_format(format, "export_documents")  # noqa: A001

    dispatch: dict[ExportFormat, Any] = {
        ExportFormat.CSV: _export_csv,
        ExportFormat.JSONL: _export_text,
        ExportFormat.JSON: _export_text,
        ExportFormat.YAML: _export_text,
        ExportFormat.TOML: _export_text,
        ExportFormat.PICKLE: _export_pickle,
        ExportFormat.JOBLIB: _export_joblib,
        ExportFormat.NUMPY: _export_numpy,
        ExportFormat.PANDAS: _export_pandas,
        ExportFormat.PARQUET: _export_parquet,
        ExportFormat.POLARS: _export_polars,
        ExportFormat.HUGGINGFACE: _export_huggingface,
        ExportFormat.MLFLOW: _export_mlflow,
    }

    exporter = dispatch.get(format)
    if exporter is None:
        raise ValueError(
            f"export_documents: unknown ExportFormat {format!r}."
            f" Supported: {[f.value for f in ExportFormat]}."
        )

    logger.info(
        "export_documents: writing %d docs to %s (format=%s).",
        len(documents),
        output_path,
        format.value,
    )

    # Ensure parent directory exists for all file-based formats
    if format not in (ExportFormat.HUGGINGFACE, ExportFormat.MLFLOW):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    exporter(
        documents,
        output_path,
        include_embedding=include_embedding,
        json_indent=json_indent,
        parquet_compression=parquet_compression,
        export_format=format,
    )

    logger.info("export_documents: wrote %s.", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


def _as_format(value: Any, caller: str) -> ExportFormat:
    """Return ``value`` as an :class:`ExportFormat` (a member or its string value)."""
    if isinstance(value, ExportFormat):
        return value
    if isinstance(value, str):
        try:
            return ExportFormat(value.strip().lower())
        except ValueError:
            raise ValueError(
                f"{caller}: unknown format {value!r}; choose from {[f.value for f in ExportFormat]}."
            ) from None
    raise TypeError(
        f"{caller}: format must be an ExportFormat or its string value, got {type(value).__name__!r}."
    )


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    """
    Write ``data`` to ``path`` atomically via a sibling ``.tmp`` file.

    Parameters
    ----------
    path : pathlib.Path
        Final destination path.
    data : bytes
        Raw bytes to write.
    """
    atomic_write_bytes(path, data)


def _atomic_write_text(
    path: pathlib.Path, text: str, encoding: str = _CSV_ENCODING
) -> None:
    """
    Write ``text`` to ``path`` atomically.

    Parameters
    ----------
    path : pathlib.Path
        Final destination path.
    text : str
        Text content to write.
    encoding : str
        Target encoding. Default: ``"utf-8"``.
    """
    _atomic_write_bytes(path, text.encode(encoding))


# ---------------------------------------------------------------------------
# Individual format exporters
# ---------------------------------------------------------------------------


def _compute_csv_fieldnames(rows):
    """
    Compute superset of all keys across all rows.

    Preserves stable ordering: identity fields first, then sorted rest.
    Prevents silent column loss when later documents have metadata
    keys absent from the first document.
    """
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    # Stable ordering: identity fields first (single source of truth at module level).
    fieldnames = [k for k in _CSV_IDENTITY_ORDER if k in all_keys]
    fieldnames += sorted(all_keys - set(fieldnames))
    return fieldnames


def _export_csv(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    **_kwargs: Any,
) -> None:
    """
    Export as flat CSV using stdlib ``csv.DictWriter``.

    Metadata dict keys are flattened into top-level columns via
    :meth:`~scikitplot.corpus._schema.CorpusDocument.to_flat_dict`.
    Embeddings are never included (not suited to tabular cells).

    When ``documents`` is empty a **header-only CSV** is written using
    :data:`_CSV_IDENTITY_ORDER` as the column list.  This guarantees that
    ``pd.read_csv(output_path)`` always succeeds and returns an empty
    ``DataFrame`` with named columns instead of raising
    ``pandas.errors.EmptyDataError``.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        Ignored for CSV — embeddings are always excluded.

    Notes
    -----
    **Developer note:** A zero-byte file was written previously for the
    empty case.  ``pandas.read_csv`` raises ``EmptyDataError`` on a
    zero-byte file.  The header-only approach is consistent with what
    ``pandas.DataFrame.to_csv`` produces for an empty ``DataFrame`` and
    lets callers differentiate "empty corpus" from "missing file".
    """
    buf = io.StringIO()

    if not documents:
        # Write identity columns only — no rows.  Callers get a parseable,
        # zero-row DataFrame rather than EmptyDataError.
        writer = csv.DictWriter(
            buf, fieldnames=_CSV_IDENTITY_ORDER, extrasaction="ignore"
        )
        writer.writeheader()
        _atomic_write_text(output_path, buf.getvalue())
        logger.debug(
            "_export_csv: no documents — wrote header-only CSV to %s.", output_path
        )
        return

    rows = [doc.to_flat_dict(include_embedding=False) for doc in documents]
    fieldnames = _compute_csv_fieldnames(rows)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(output_path, buf.getvalue())


def document_records(
    documents: list[CorpusDocument], *, include_embedding: bool, fmt: str = "json"
) -> list[dict[str, Any]]:
    """
    Return each document as a record of the serialisation data model.

    Parameters
    ----------
    documents : list of CorpusDocument
        Documents to convert.
    include_embedding : bool
        Include ``embedding`` (a list of floats) and ``embedding_dtype``.
    fmt : {"json", "jsonl", "yaml", "toml"}, default="json"
        Format the records are checked against
        (:func:`._records.export_problem`).

    Returns
    -------
    list of dict

    Raises
    ------
    TypeError
        If a value is not serialisable; the message names the document.
    ValueError
        If a value cannot be represented in ``fmt``, the embedding cannot be
        written exactly (not one-dimensional, or a dtype outside
        :data:`._records.EMBEDDING_DTYPES`), or the metadata is nested too
        deeply; the message names the document.
    """
    records = []
    for index, doc in enumerate(documents):
        label = f"export_documents ({fmt}): document {index} (doc_id={doc.doc_id!r})"
        record = doc.to_dict(include_embedding=False)
        if include_embedding and doc.embedding is not None:
            try:
                record[EMBEDDING_DTYPE_KEY] = embedding_dtype(doc.embedding)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}: {exc}.") from None
            record["embedding"] = [float(v) for v in np.asarray(doc.embedding).tolist()]
        try:
            record = _serialize.to_builtin(record)
        except TypeError as exc:
            raise TypeError(f"{label} cannot be exported: {exc}") from None
        except RecursionError:
            raise ValueError(
                f"{label}: the metadata is nested too deeply to export."
            ) from None
        problem = export_problem(record, fmt)
        if problem:
            raise ValueError(f"{label}: {problem}.")
        records.append(record)
    return records


def _export_text(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    json_indent: int | None,
    export_format: ExportFormat,
    **_kwargs: Any,
) -> None:
    """
    Export as JSON, JSON Lines, YAML or TOML through :mod:`.._serialize`.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        When ``True``, embedding vectors are written as lists of floats with
        their dtype name.
    json_indent : int or None
        Indentation for JSON; JSON Lines is always one compact line per
        document; other formats ignore it.
    export_format : ExportFormat
        One of :data:`TEXT_FORMATS`.

    Notes
    -----
    **Developer note.** :func:`document_records` checks each record before
    anything is written. A serialiser limit it does not model (an integer
    with more digits than Python converts to text) is found after the fact:
    each record is serialised alone until the failing one is named.
    """
    fmt = TEXT_FORMATS[export_format]
    records = document_records(documents, include_embedding=include_embedding, fmt=fmt)
    payload: Any = {TOML_DOCUMENTS_KEY: records} if fmt == "toml" else records
    try:
        _serialize.write(output_path, payload, fmt, indent=json_indent)
    except (ValueError, UnicodeError, RecursionError) as exc:
        for index, record in enumerate(records):
            try:
                _serialize.dumps(
                    {TOML_DOCUMENTS_KEY: [record]} if fmt == "toml" else [record], fmt
                )
            except (  # noqa: PERF203 - failure path only
                ValueError,
                UnicodeError,
                RecursionError,
            ) as record_exc:
                raise ValueError(
                    f"export_documents ({fmt}): document {index} (doc_id={record.get('doc_id')!r}) "
                    f"cannot be written: {record_exc}"
                ) from exc
        raise


def _export_jsonl(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    **_kwargs: Any,
) -> None:
    """Export as JSON Lines (used by the MLflow exporter)."""
    _export_text(
        documents,
        output_path,
        include_embedding=include_embedding,
        json_indent=None,
        export_format=ExportFormat.JSONL,
    )


def _export_pickle(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    **_kwargs: Any,
) -> None:
    """
    Export via ``pickle.dump``.

    Preserves full Python objects including embeddings and metadata.
    Useful for checkpointing intermediate pipeline results.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        When ``False``, embeddings are stripped before pickling to
        reduce file size.
    """
    objs = documents
    if not include_embedding:
        objs = [doc.replace(embedding=None) for doc in documents]

    data = pickle.dumps(objs, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write_bytes(output_path, data)


def _export_joblib(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    **_kwargs: Any,
) -> None:
    """
    Export via ``joblib.dump``.

    Faster than pickle for documents with large NumPy embedding arrays.
    Uses joblib's memory-mapped array serialisation.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool

    Raises
    ------
    ImportError
        If ``joblib`` is not installed.
    """
    try:
        import joblib  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "joblib is required for ExportFormat.JOBLIB."
            " Install it with:\n"
            "  pip install joblib"
        ) from exc

    objs = documents
    if not include_embedding:
        objs = [doc.replace(embedding=None) for doc in documents]

    atomic_write_path(output_path, lambda p: joblib.dump(objs, p))


def _export_numpy(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    **_kwargs: Any,
) -> None:
    """
    Export embedding matrix as a ``.npy`` file.

    Shape: ``(n_documents, embedding_dim)``.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        Ignored — embeddings are always written for this format.

    Raises
    ------
    ValueError
        If no documents carry embeddings.
    ValueError
        If embedding dimensions are inconsistent across documents.
    """
    embedded = [doc for doc in documents if doc.has_embedding]
    if not embedded:
        raise ValueError(
            "export_documents (NUMPY): no documents have embeddings."
            " Run EmbeddingEngine.embed_documents() before exporting."
        )

    dims = {doc.embedding.shape[0] for doc in embedded}
    if len(dims) > 1:
        raise ValueError(
            f"export_documents (NUMPY): inconsistent embedding dimensions"
            f" {dims}. All documents must have the same embedding size."
        )

    matrix = np.vstack([doc.embedding for doc in embedded]).astype(
        np.float32, copy=False
    )

    # np.save() appends ".npy" automatically when the path does not already
    # end with it.  Use a temp path that already ends with ".npy" so that
    atomic_write_path(
        output_path,
        lambda p: np.save(str(p), matrix, allow_pickle=False),
        suffix=".npy",
    )

    logger.debug(
        "export_documents (NUMPY): saved matrix shape %s dtype %s.",
        matrix.shape,
        matrix.dtype,
    )


def _export_pandas(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    **_kwargs: Any,
) -> None:
    """
    Export as a pandas CSV (via ``DataFrame.to_csv``).

    Richer dtype handling than the stdlib CSV exporter. Embeddings are
    excluded regardless of ``include_embedding`` (not tabular-friendly).

    When ``documents`` is empty a **header-only CSV** is written using
    :data:`_CSV_IDENTITY_ORDER` as the column list.  This mirrors the
    behaviour of :func:`_export_csv` and prevents
    ``pandas.errors.EmptyDataError`` on the read side.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        Ignored — embeddings are always excluded from tabular output.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed.

    Notes
    -----
    **Developer note:** ``pd.DataFrame([]).to_csv(...)`` writes only a
    trailing newline (no header), which also triggers ``EmptyDataError``.
    Constructing the empty DataFrame with explicit ``columns`` writes a
    proper header row.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pandas is required for ExportFormat.PANDAS."
            " Install it with:\n"
            "  pip install pandas"
        ) from exc

    if not documents:
        # Build a header-only DataFrame — columns defined, no rows.
        # pd.DataFrame([]) would produce no columns and therefore no
        # header, triggering EmptyDataError on re-read.
        df = pd.DataFrame(columns=_CSV_IDENTITY_ORDER)
        logger.debug(
            "_export_pandas: no documents — wrote header-only CSV to %s.", output_path
        )
    else:
        rows = [doc.to_pandas_row(include_embedding=False) for doc in documents]
        df = pd.DataFrame(rows)

    atomic_write_path(
        output_path,
        lambda p: df.to_csv(p, index=False, encoding=_CSV_ENCODING),
    )


def _export_parquet(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    parquet_compression: str = "snappy",
    **_kwargs: Any,
) -> None:
    """
    Export as Apache Parquet via ``pandas.DataFrame.to_parquet``.

    Requires ``pyarrow`` or ``fastparquet`` in addition to ``pandas``.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        Ignored — embeddings excluded from tabular output.
    parquet_compression : str
        Compression codec. Default: ``"snappy"``.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed or no Parquet engine is found.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pandas is required for ExportFormat.PARQUET."
            " Install it with:\n"
            "  pip install pandas pyarrow"
        ) from exc

    rows = [doc.to_pandas_row(include_embedding=False) for doc in documents]
    df = pd.DataFrame(rows)

    atomic_write_path(
        output_path,
        lambda p: df.to_parquet(p, index=False, compression=parquet_compression),
    )


def _export_polars(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    parquet_compression: str = "snappy",
    **_kwargs: Any,
) -> None:
    """
    Export as Apache Parquet via ``polars.DataFrame.write_parquet``.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        Ignored — embeddings excluded from tabular output.
    parquet_compression : str
        Compression codec passed to polars. Default: ``"snappy"``.

    Raises
    ------
    ImportError
        If ``polars`` is not installed.
    """
    try:
        import polars as pl  # type: ignore[] # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "polars is required for ExportFormat.POLARS."
            " Install it with:\n"
            "  pip install polars"
        ) from exc

    rows = [doc.to_polars_row(include_embedding=False) for doc in documents]
    df = pl.DataFrame(rows)

    atomic_write_path(
        output_path,
        lambda p: df.write_parquet(p, compression=parquet_compression),
    )


def _export_huggingface(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    **_kwargs: Any,
) -> None:
    """
    Export as a HuggingFace ``datasets.Dataset`` saved to disk.

    Produces an Arrow-format dataset directory tree rooted at
    ``output_path``. Load it back with
    ``datasets.load_from_disk(output_path)``.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
        Target directory (created if absent).
    include_embedding : bool
        When ``True``, embeddings are stored as a ``"embedding"``
        ``Sequence(Value("float32"))`` feature.

    Raises
    ------
    ImportError
        If ``datasets`` is not installed.
    """
    try:
        import datasets  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "datasets is required for ExportFormat.HUGGINGFACE."
            " Install it with:\n"
            "  pip install datasets"
        ) from exc

    # Build feature dict; only include embeddings if all docs have them
    records: list[dict[str, Any]] = []
    for doc in documents:
        row = doc.to_dict(include_embedding=include_embedding)
        if "embedding" in row and row["embedding"] is not None:
            emb = row["embedding"]
            if hasattr(emb, "tolist"):
                row["embedding"] = emb.tolist()
        # Flatten metadata into top-level keys
        meta = row.pop("metadata", {}) or {}
        row.update(meta)
        records.append(row)

    ds = datasets.Dataset.from_list(records)
    output_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path))
    logger.debug(
        "export_documents (HUGGINGFACE): saved %d rows to %s.",
        len(records),
        output_path,
    )


def _export_mlflow(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    **_kwargs: Any,
) -> None:
    """
    Log documents as an MLflow artifact inside the active run.

    Writes documents to a temporary JSONL file, then logs it as an
    MLflow artifact under the ``output_path`` prefix (artifact path).

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
        Used as the MLflow ``artifact_path`` prefix (directory within
        the run's artifact store).
    include_embedding : bool
        When ``True``, embeddings are included in the logged JSON.

    Raises
    ------
    ImportError
        If ``mlflow`` is not installed.
    RuntimeError
        If no active MLflow run is found. Call ``mlflow.start_run()``
        before exporting.

    Notes
    -----
    The actual artifact file is named ``corpus_documents.jsonl`` and is
    placed under the ``output_path`` artifact prefix.
    """
    try:
        import mlflow  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "mlflow is required for ExportFormat.MLFLOW."
            " Install it with:\n"
            "  pip install mlflow"
        ) from exc

    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError(
            "export_documents (MLFLOW): no active MLflow run."
            " Call mlflow.start_run() before exporting."
        )

    import tempfile  # noqa: PLC0415

    artifact_path = str(output_path) if str(output_path) != "." else None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = pathlib.Path(tmpdir) / "corpus_documents.jsonl"
        _export_jsonl(
            documents,
            tmp_file,
            include_embedding=include_embedding,
            json_indent=None,
        )
        mlflow.log_artifact(str(tmp_file), artifact_path=artifact_path)

    logger.info(
        "export_documents (MLFLOW): logged %d docs as artifact (run=%s).",
        len(documents),
        active_run.info.run_id,
    )


# ---------------------------------------------------------------------------
# Convenience loader — symmetric round-trip for pickle and joblib
# ---------------------------------------------------------------------------


def _pickle_safety_guard(format_value, trusted):
    """Insert this check before pickle.load() and joblib.load() calls.

    Parameters
    ----------
    format_value : str
        The format value string (e.g., "pickle", "joblib").
    trusted : bool
        Whether the user has explicitly opted in to unsafe loading.

    Raises
    ------
    ValueError
        If trusted is False and format is pickle or joblib.
    """
    if format_value in ("pickle", "joblib") and not trusted:
        raise ValueError(
            f"Loading {format_value} files is disabled by default due to "
            f"arbitrary code execution risk. A malicious {format_value} file "
            f"can execute arbitrary Python code on your system.\n"
            f"Pass trusted=True to load_documents() ONLY if you trust "
            f"the source of this file.\n"
            f"Safer alternatives: use Parquet or JSON format."
        )


_INTEGRITY_CHUNK = 1024 * 1024  # 1 MiB


def _verify_artifact_integrity(path, expected_sha256):
    """Verify a file's SHA-256 BEFORE deserialization (CORPUS-SEC-001).

    When ``expected_sha256`` is provided, the artifact bytes are hashed and
    compared *before* any ``pickle.load`` runs, so a tampered or wrong artifact
    is rejected before it can execute code.

    Parameters
    ----------
    path : pathlib.Path
        Artifact to verify.
    expected_sha256 : str or None
        Expected lowercase hex SHA-256. ``None`` skips the check.

    Raises
    ------
    ValueError
        If the actual digest does not match ``expected_sha256``.
    """
    if expected_sha256 is None:
        return
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_INTEGRITY_CHUNK), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected_sha256.strip().lower():
        raise ValueError(
            f"load_documents: {str(path)!r} SHA-256 {actual} does not match "
            f"the expected {expected_sha256!r}. Refusing to load a tampered or "
            f"wrong artifact."
        )


def _validate_loaded_documents(obj, path):
    """Validate a deserialized object matches the documented return contract.

    A trusted-but-wrong artifact must fail loudly rather than return an
    arbitrary object shape (CORPUS-SEC-001). Code execution during ``pickle``
    is inherent, but the *result* is still held to the ``list[CorpusDocument]``
    contract.

    Parameters
    ----------
    obj : Any
        The deserialized object.
    path : pathlib.Path
        Source path, for error messages.

    Returns
    -------
    list of CorpusDocument

    Raises
    ------
    TypeError
        If ``obj`` is not a list of :class:`CorpusDocument`.
    """
    if not isinstance(obj, list):
        raise TypeError(
            f"load_documents: {str(path)!r} deserialized to "
            f"{type(obj).__name__}, expected list[CorpusDocument]. Refusing to "
            f"return an unexpected object shape."
        )
    for index, item in enumerate(obj):
        if not isinstance(item, CorpusDocument):
            raise TypeError(
                f"load_documents: {str(path)!r} element {index} is "
                f"{type(item).__name__}, expected CorpusDocument."
            )
    return obj


def _document_from_record(
    record: Any, index: int, path: pathlib.Path
) -> CorpusDocument:
    """Rebuild one document from a text-format record, refusing anything off contract."""
    label = f"load_documents: {str(path)!r} record {index}"
    problem = record_problem(record)
    if problem:
        raise ValueError(f"{label}: {problem}.")
    data = dict(record)
    dtype_name = data.pop(EMBEDDING_DTYPE_KEY, None)
    try:
        if data.get("embedding") is not None:
            data["embedding"] = embedding_array(
                data["embedding"], dtype_name or "float64"
            )
        return CorpusDocument.from_dict(data)
    except (TypeError, ValueError, KeyError, AttributeError, ArithmeticError) as exc:
        raise ValueError(f"{label}: {exc}") from exc


def _load_text(path: pathlib.Path, export_format: ExportFormat) -> list[CorpusDocument]:
    """Read a JSON, JSON Lines, YAML or TOML export."""
    fmt = TEXT_FORMATS[export_format]
    try:
        data = _serialize.read(path, fmt)
    except ValueError as exc:  # parse errors, NaN tokens, aliases, encoding errors
        hint = (
            " (earlier versions of export_documents could write NaN tokens; re-export the corpus)"
            if "NaN" in str(exc)
            else ""
        )
        raise ValueError(
            f"load_documents: {str(path)!r} is not valid {fmt}: {exc}{hint}"
        ) from exc
    if fmt == "toml":
        if not isinstance(data, Mapping) or set(data) != {TOML_DOCUMENTS_KEY}:
            raise ValueError(
                f"load_documents: {str(path)!r} is not a document export: a TOML export has "
                f"exactly one key, {TOML_DOCUMENTS_KEY!r}."
            )
        data = data[TOML_DOCUMENTS_KEY]
    if not isinstance(data, list):
        raise ValueError(  # noqa: TRY004 - malformed file content, as json.loads reports it
            f"load_documents: {str(path)!r} holds {type(data).__name__}, expected a list of documents."
        )
    return [
        _document_from_record(record, index, path) for index, record in enumerate(data)
    ]


def load_documents(
    path: pathlib.Path | str,
    format: ExportFormat | None = None,  # noqa: A002
    *,
    trusted: bool = False,
    expected_sha256: str | None = None,
) -> list[CorpusDocument]:
    """
    Load :class:`~scikitplot.corpus._schema.CorpusDocument` instances
    from a previously exported file.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the exported file.
    format : ExportFormat, str or None, optional
        The export format, or its string value (``"jsonl"``). When ``None``,
        it is inferred from the suffix:
        ``.json``, ``.jsonl``/``.ndjson``, ``.yaml``/``.yml``, ``.toml``,
        ``.pkl``/``.pickle``, ``.joblib``.
    trusted : bool, default=False
        Required for pickle and joblib, whose loading can execute code.
        JSON, JSON Lines, YAML and TOML never need it.
    expected_sha256 : str or None, optional
        When provided, the file's SHA-256 is verified **before** it is
        parsed or deserialized; a mismatch raises :class:`ValueError`.

    Returns
    -------
    list of CorpusDocument
        In file order. Embeddings come back as NumPy arrays of the recorded
        ``embedding_dtype`` (``float64`` when a record names none).

    Raises
    ------
    ValueError
        If the format cannot be inferred or cannot be loaded back (CSV,
        pandas, Parquet, polars, NumPy, HuggingFace; for MLflow, load the
        downloaded ``corpus_documents.jsonl``); if loading pickle or joblib
        is not ``trusted``; if ``expected_sha256`` does not match; if the
        file is not valid in its format (the message names the file); or if
        a record breaks the record contract: an unknown key, a field of the
        wrong type, a YAML or TOML date where a string belongs, an embedding
        value that overflows its dtype (the message names the record).
    ImportError
        If the library a format needs is not installed.
    OSError
        If the file cannot be read.
    TypeError
        If a trusted pickle or joblib file does not hold a list of
        :class:`~scikitplot.corpus._schema.CorpusDocument`.

    See Also
    --------
    export_documents : The writer.

    Notes
    -----
    **User note.** Text formats are the safe way to keep a corpus: loading
    runs no code, the files are diffable, and every field the exporter wrote
    comes back. Tabular and directory formats cannot rebuild every field
    (metadata is flattened, embeddings are dropped), so loading them is
    refused instead of returning a partial or empty result.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> from scikitplot.corpus import CorpusDocument, ExportFormat, export_documents
    >>> doc = CorpusDocument.create("notes.txt", 0, "Hello.")
    >>> out = export_documents(
    ...     [doc], Path(tempfile.mkdtemp()) / "corpus.jsonl", ExportFormat.JSONL
    ... )
    >>> [d.text for d in load_documents(out)]
    ['Hello.']

    Loading a pickle/joblib export requires an explicit trust decision, because
    deserialization can execute arbitrary code. Pin a known digest when you
    have one:

    >>> docs = load_documents(  # doctest: +SKIP
    ...     Path("corpus.pkl"),
    ...     trusted=True,
    ...     expected_sha256="e3b0c44298fc1c149afbf4c8996fb924...",
    ... )
    """  # noqa: D205
    path = pathlib.Path(path)
    if format is None:
        format = LOAD_SUFFIXES.get(path.suffix.lower())  # noqa: A001
        if format is None:
            raise ValueError(
                f"load_documents: cannot infer the format of {str(path)!r} from its suffix; "
                f"pass format= (loadable: {[f.value for f in (*TEXT_FORMATS, ExportFormat.PICKLE, ExportFormat.JOBLIB)]})."
            )
    format = _as_format(format, "load_documents")  # noqa: A001
    if format == ExportFormat.MLFLOW:
        raise ValueError(
            "load_documents: an MLflow export is a JSON Lines artifact of a run; download "
            "corpus_documents.jsonl from the run and load that file (format jsonl)."
        )
    if format not in TEXT_FORMATS and format not in (
        ExportFormat.PICKLE,
        ExportFormat.JOBLIB,
    ):
        raise ValueError(
            f"load_documents: {format.value!r} exports cannot be loaded back as documents "
            "(they do not keep every field); export as json, jsonl, yaml or toml to load them."
        )
    _pickle_safety_guard(format.value, trusted)
    # Integrity gate runs BEFORE parsing or deserialization: a tampered/wrong
    # artifact is rejected before pickle can execute code (CORPUS-SEC-001).
    _verify_artifact_integrity(path, expected_sha256)

    if format in TEXT_FORMATS:
        return _load_text(path, format)

    if format == ExportFormat.PICKLE:
        with path.open("rb") as fh:
            obj = pickle.load(fh)  # noqa: S301
        return _validate_loaded_documents(obj, path)

    try:
        import joblib  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "joblib is required to load a JOBLIB export."
            " Install it with:\n"
            "  pip install joblib"
        ) from exc
    return _validate_loaded_documents(joblib.load(path), path)
