# scikitplot/corpus/_export/_export.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._export
==========================
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
    One JSON object per line (newline-delimited JSON). Zero optional
    dependencies. Metadata dict is preserved as a nested object.

:attr:`ExportFormat.JSON`
    JSON array of document dicts. Zero optional dependencies. Indented
    by default for readability.

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

Python compatibility
--------------------
Python 3.8-3.15. Only ``csv``, ``json``, and ``pickle`` (all stdlib)
and ``numpy`` are hard requirements. All other backends are optional.
"""  # noqa: D205, D400

from __future__ import annotations

import csv
import io
import json
import logging
import pathlib
import pickle
from typing import Any, Dict, List, Optional  # noqa: F401

import numpy as np

from .._schema import CorpusDocument, ExportFormat

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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def export_documents(
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    fmt: ExportFormat,
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
    fmt : ExportFormat
        Target export format.
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
        If ``fmt`` is :attr:`ExportFormat.NUMPY` and no documents have
        embeddings, or if the embedding dimensions are inconsistent.
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
    impractical. Use PICKLE, JOBLIB, or NUMPY to preserve embeddings.

    Examples
    --------
    CSV export (zero dependencies):

    >>> from pathlib import Path
    >>> export_documents(docs, Path("corpus.csv"), ExportFormat.CSV)
    PosixPath('corpus.csv')

    JSONL with embeddings:

    >>> export_documents(
    ...     docs,
    ...     Path("corpus.jsonl"),
    ...     ExportFormat.JSONL,
    ...     include_embedding=True,
    ... )

    NumPy embedding matrix:

    >>> export_documents(docs, Path("embeddings.npy"), ExportFormat.NUMPY)
    PosixPath('embeddings.npy')
    """
    output_path = pathlib.Path(output_path)

    dispatch: dict[ExportFormat, Any] = {
        ExportFormat.CSV: _export_csv,
        ExportFormat.JSONL: _export_jsonl,
        ExportFormat.JSON: _export_json,
        ExportFormat.PICKLE: _export_pickle,
        ExportFormat.JOBLIB: _export_joblib,
        ExportFormat.NUMPY: _export_numpy,
        ExportFormat.PANDAS: _export_pandas,
        ExportFormat.PARQUET: _export_parquet,
        ExportFormat.POLARS: _export_polars,
        ExportFormat.HUGGINGFACE: _export_huggingface,
        ExportFormat.MLFLOW: _export_mlflow,
    }

    exporter = dispatch.get(fmt)
    if exporter is None:
        raise ValueError(
            f"export_documents: unknown ExportFormat {fmt!r}."
            f" Supported: {[f.value for f in ExportFormat]}."
        )

    logger.info(
        "export_documents: writing %d docs to %s (format=%s).",
        len(documents),
        output_path,
        fmt.value,
    )

    # Ensure parent directory exists for all file-based formats
    if fmt not in (ExportFormat.HUGGINGFACE, ExportFormat.MLFLOW):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    exporter(
        documents,
        output_path,
        include_embedding=include_embedding,
        json_indent=json_indent,
        parquet_compression=parquet_compression,
    )

    logger.info("export_documents: wrote %s.", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Atomic write helpers
# ---------------------------------------------------------------------------


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
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


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

    # Stable ordering: identity fields first
    identity_order = [
        "doc_id",
        "input_path",
        "chunk_index",
        "text",
        "section_type",
        "chunking_strategy",
        "language",
        "source_type",
        "source_title",
        "source_author",
    ]
    fieldnames = [k for k in identity_order if k in all_keys]
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

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        Ignored for CSV — embeddings are always excluded.
    """
    if not documents:
        _atomic_write_text(output_path, "")
        return

    rows = [doc.to_flat_dict(include_embedding=False) for doc in documents]
    fieldnames = _compute_csv_fieldnames(rows)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write_text(output_path, buf.getvalue())


def _export_jsonl(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    json_indent: int | None,
    **_kwargs: Any,
) -> None:
    """
    Export as newline-delimited JSON (one object per line).

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
        When ``True``, embedding vectors are serialised as JSON arrays.
    json_indent : int or None
        Indentation per line. ``None`` produces compact single-line JSON.
        For JSONL, ``None`` is strongly recommended (each line must be
        exactly one JSON object).
    """
    lines = []
    for doc in documents:
        d = doc.to_dict(include_embedding=include_embedding)
        # to_dict() already converts the embedding to a Python list;
        # ensure any residual ndarray is serialised safely
        if "embedding" in d and d["embedding"] is not None:
            emb = d["embedding"]
            if hasattr(emb, "tolist"):
                d["embedding"] = emb.tolist()
        lines.append(json.dumps(d, indent=None))

    _atomic_write_text(output_path, "\n".join(lines) + ("\n" if lines else ""))


def _export_json(  # noqa: D417
    documents: list[CorpusDocument],
    output_path: pathlib.Path,
    *,
    include_embedding: bool,
    json_indent: int | None,
    **_kwargs: Any,
) -> None:
    """
    Export as a JSON array.

    Parameters
    ----------
    documents : list of CorpusDocument
    output_path : pathlib.Path
    include_embedding : bool
    json_indent : int or None
        Indentation for the JSON output. Default: 2.
    """
    records = []
    for doc in documents:
        d = doc.to_dict(include_embedding=include_embedding)
        if "embedding" in d and d["embedding"] is not None:
            emb = d["embedding"]
            if hasattr(emb, "tolist"):
                d["embedding"] = emb.tolist()
        records.append(d)

    text = json.dumps(records, indent=json_indent, ensure_ascii=False)
    _atomic_write_text(output_path, text)


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

    tmp = output_path.with_name(output_path.name + ".tmp")
    try:
        joblib.dump(objs, tmp)
        tmp.replace(output_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


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
    # np.save writes exactly the file we expect, then rename atomically.
    tmp = output_path.with_name(output_path.stem + ".tmp.npy")
    try:
        np.save(str(tmp), matrix, allow_pickle=False)
        tmp.replace(output_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

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
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pandas is required for ExportFormat.PANDAS."
            " Install it with:\n"
            "  pip install pandas"
        ) from exc

    rows = [doc.to_pandas_row(include_embedding=False) for doc in documents]
    df = pd.DataFrame(rows)

    tmp = output_path.with_name(output_path.name + ".tmp")
    try:
        df.to_csv(tmp, index=False, encoding=_CSV_ENCODING)
        tmp.replace(output_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


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

    tmp = output_path.with_name(output_path.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False, compression=parquet_compression)
        tmp.replace(output_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


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

    tmp = output_path.with_name(output_path.name + ".tmp")
    try:
        df.write_parquet(tmp, compression=parquet_compression)
        tmp.replace(output_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


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


def _pickle_safety_guard(fmt_value, trusted):
    """Insert this check before pickle.load() and joblib.load() calls.

    Parameters
    ----------
    fmt_value : str
        The format value string (e.g., "pickle", "joblib").
    trusted : bool
        Whether the user has explicitly opted in to unsafe loading.

    Raises
    ------
    ValueError
        If trusted is False and format is pickle or joblib.
    """
    if fmt_value in ("pickle", "joblib") and not trusted:
        raise ValueError(
            f"Loading {fmt_value} files is disabled by default due to "
            f"arbitrary code execution risk. A malicious {fmt_value} file "
            f"can execute arbitrary Python code on your system.\n"
            f"Pass trusted=True to load_documents() ONLY if you trust "
            f"the source of this file.\n"
            f"Safer alternatives: use Parquet or JSON format."
        )


def load_documents(
    path: pathlib.Path | str,
    fmt: ExportFormat | None = None,
    *,
    trusted: bool = False,
) -> list[CorpusDocument]:
    """
    Load :class:`~scikitplot.corpus._schema.CorpusDocument` instances
    from a previously exported file.

    Supported round-trip formats: :attr:`ExportFormat.PICKLE`,
    :attr:`ExportFormat.JOBLIB`. For all other formats, returns an empty
    list with a warning (full deserialization from CSV/JSON/Parquet is
    handled separately by the pipeline).

    Parameters
    ----------
    path : pathlib.Path
        Path to the exported file.
    fmt : ExportFormat or None, optional
        Format hint. When ``None``, the format is inferred from the
        file extension (``.pkl`` → PICKLE, ``.joblib`` → JOBLIB).
    trusted : bool
        Whether the user has explicitly opted in to unsafe loading.

    Returns
    -------
    list of CorpusDocument

    Raises
    ------
    ImportError
        If ``joblib`` is not installed and the file is a joblib dump.
    OSError
        If the file cannot be read.

    Examples
    --------
    >>> docs = load_documents(Path("corpus.pkl"))
    >>> len(docs)
    312
    """  # noqa: D205
    path = pathlib.Path(path)

    if fmt is None:
        ext = path.suffix.lower()
        fmt = {
            ".pkl": ExportFormat.PICKLE,
            ".joblib": ExportFormat.JOBLIB,
        }.get(ext)
    _pickle_safety_guard(fmt.value if fmt else "", trusted)

    if fmt == ExportFormat.PICKLE:
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301

    if fmt == ExportFormat.JOBLIB:
        try:
            import joblib  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "joblib is required to load a JOBLIB export."
                " Install it with:\n"
                "  pip install joblib"
            ) from exc
        return joblib.load(path)

    logger.warning(
        "load_documents: format %r does not support CorpusDocument"
        " deserialization. Returning empty list.",
        fmt,
    )
    return []
