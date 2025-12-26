# scikitplot/annoy/_mixins/_ndarray.py
"""
Vector export & interoperability mixin for Annoy-like indexes.

This module defines :class:`NDArrayExportMixin`, a high-level export mixin focused
on deterministic, explicit transformations from (ids -> vectors) into common
ecosystem formats used in data science and MLOps:

- NumPy: in-memory arrays and on-disk `.npy` (memmap), `.npz`
- Pandas: DataFrame materialization (small/medium)
- CSV/TSV: streaming, row-oriented export (large-friendly)
- Parquet: streaming, columnar export (analytics-friendly; requires pyarrow)
- SQLite: streaming export for DB interoperability
- SciPy: convenience sparse CSR conversion (requires scipy)
- MLflow: log export artifacts + stable metadata

Big-picture constraints / non-goals
-----------------------------------
- This mixin never mutates the underlying Annoy index.
- No hidden sampling, no implicit truncation, no “auto” behavior that changes
  semantics based on size. If something requires the number of rows (pre-allocation),
  the API demands a sized ids sequence or `n_rows=...`.
- Optional dependencies are imported lazily: importing this module does not
  require pandas/pyarrow/scipy/mlflow.

Required Annoy-like surface
---------------------------
The consuming class MUST provide:
- `get_item_vector(i: int) -> Sequence[float]`
- `get_n_items() -> int`
- attribute/property `f` (dimension; may be 0 for lazy/uninitialized)

See Also
--------
scikitplot.cexternals._annoy._plotting.annoy_index_to_array
    Similar goal (materialize vectors), used by visualization helpers.
scikitplot.annoy._mixins._manifest
    Manifest/metadata helpers (if exposed in your high-level API).
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from typing import Any, Iterable, Iterator, Mapping, Sequence, Union

from typing_extensions import Literal, TypeAlias

# --- Optional deps ------------------------------------------------------------
try:  # pragma: no cover
    import numpy as np
    from numpy.lib.format import open_memmap
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    open_memmap = None  # type: ignore[assignment]

try:  # pragma: no cover
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

PathLikeStr: TypeAlias = Union[str, os.PathLike[str]]
IdsInput: TypeAlias = Union[Sequence[int], Iterable[int], None]

__all__: tuple[str] = ("NDArrayExportMixin",)


def _fspath(path: PathLikeStr) -> str:
    """Return a filesystem path as `str` (supports Path-like objects)."""
    return os.fspath(path)


class NDArrayExportMixin:
    """
    Export/interoperability utilities for Annoy-like objects.

    Notes
    -----
    - Methods that must pre-allocate (e.g. `to_numpy`, `save_vectors_npy`) require
      `ids` to be a sized sequence OR require `n_rows=...` explicitly.
    - Streaming writers (`to_csv`, `to_parquet`, `to_sqlite`) accept any iterable
      of ids (including generators).
    """

    # Provide a stable path-like coercion hook for all mixin methods.
    # (Module-level `_fspath` remains for backwards compatibility.)
    @staticmethod
    def _fspath(path: PathLikeStr) -> str:
        return os.fspath(path)

    # ------------------------------------------------------------------ #
    # Dependency guards
    # ------------------------------------------------------------------ #
    @staticmethod
    def _require_numpy() -> None:
        if np is None or open_memmap is None:
            raise RuntimeError("NumPy is required for this operation.")

    @staticmethod
    def _require_pandas() -> None:
        if pd is None:
            raise RuntimeError("pandas is required for this operation.")

    # ------------------------------------------------------------------ #
    # Internal helpers (strict, deterministic)
    # ------------------------------------------------------------------ #

    def _normalize_range(
        self,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[int, int]:
        if start < 0:
            raise ValueError("start must be >= 0")
        n_items = int(self.get_n_items())
        if stop is None:
            stop = n_items
        if stop < start:
            raise ValueError("stop must be >= start")
        stop = min(int(stop), n_items)
        return int(start), int(stop)

    def _normalize_ids(
        self,
        ids: IdsInput,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[Iterable[int], int]:
        """
        Return (iterable_ids, n_rows).

        Strict sizing rules:
        - If ids is provided:
            - must be a Sequence (has __len__) OR
            - user must provide explicit sized iterable elsewhere (not supported here)
        - If ids is None:
            - we use range(start, stop) and can compute size.
        """
        if ids is None:
            s, e = self._normalize_range(start=start, stop=stop)
            return range(s, e), (e - s)

        # ids provided
        if isinstance(ids, Sequence):
            n_rows = len(ids)
            return ids, n_rows

        # Iterable but not sized -> strict error
        raise TypeError(
            "ids must be a sized Sequence when provided. "
            "For very large exports, pass a range or a concrete list/array."
        )

    def _iter_ids(
        self, ids: IdsInput, start: int = 0, stop: int | None = None
    ) -> Iterator[int]:
        """
        Return an iterator of ids.

        Compatibility rule:
        - If `ids` is provided, `start/stop` are ignored (same as previous behavior).
        - If `ids` is None, iterate range(start, stop) with stop defaulting to n_items.
        """
        if ids is not None:
            return (int(i) for i in ids)

        n_items = int(self.get_n_items())
        s = int(start)
        if s < 0:
            raise ValueError("`start` must be >= 0.")
        e = n_items if stop is None else int(stop)
        if e < s:
            raise ValueError("`stop` must be >= `start`.")
        e = min(e, n_items)
        return iter(range(s, e))

    def _n_rows(
        self,
        ids: IdsInput,
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
    ) -> int:
        """
        Determine number of rows for exports that require pre-allocation.

        Rules:
        - ids is None -> infer from range(start, stop)
        - ids is a sized Sequence -> len(ids)
        - ids is a non-sized Iterable -> require n_rows explicitly
        """
        if ids is None:
            n_items = int(self.get_n_items())
            s = int(start)
            e = n_items if stop is None else int(stop)
            e = min(e, n_items)
            return max(0, e - s)

        if isinstance(ids, Sequence):
            return len(ids)

        if n_rows is None:
            raise TypeError(
                "`ids` must be a sized sequence or you must pass `n_rows=...` explicitly."
            )
        if int(n_rows) < 0:
            raise ValueError("`n_rows` must be >= 0.")
        return int(n_rows)

    def _iter_ids_with_first_vector(
        self,
        ids: IdsInput = None,
        start: int = 0,
        stop: int | None = None,
    ) -> tuple[Iterator[int], int | None, Sequence[float] | None]:
        """
        Create an id iterator and (optionally) fetch the first vector once.

        This helper exists to keep streaming exports *single-pass*.

        It returns an iterator positioned **after** the first id (if any) plus
        the first ``(id, vector)`` pair. Callers must write/consume that first
        row explicitly before continuing to iterate.

        Notes
        -----
        - If ``stop <= start`` or the selected id set is empty, ``first_id`` and
          ``first_vec`` are ``None`` and the returned iterator is already exhausted.
        - This helper never rewinds ``ids``. If you pass a one-shot iterator
          (e.g., a generator), the consumption is intentional and explicit.
        """
        it = self._iter_ids(ids, start=start, stop=stop)
        try:
            first_id = next(it)
        except StopIteration:
            return it, None, None
        return it, int(first_id), self.get_item_vector(int(first_id))

    def _infer_f_from_first_vec(self, first_vec: Sequence[float] | None) -> int:
        """
        Return the vector dimension ``f`` from configuration or a fetched vector.

        This method is deterministic and *does not* attempt to consume ``ids``.
        """
        f = int(getattr(self, "f", 0) or 0)
        if f > 0:
            return f
        if first_vec is None:
            return 0
        return len(first_vec)

    def _materialize_dense(  # noqa: PLR0912
        self,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",  # type: ignore[]
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        validate_vector_len: bool = True,
        include_ids: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:  # type: ignore[]
        """
        Materialize vectors into a dense NumPy array, optionally returning ids.

        This is the core implementation for :meth:`to_numpy`, :meth:`to_dataframe`,
        and :meth:`save_vectors_npz`. It is designed to be:

        - **Deterministic** (row order is defined by ``ids`` / ``start`` / ``stop``)
        - **Single-pass** (works with generators and other one-shot iterables)
        - **Strict** (never silently truncates or pads when ``n_rows`` is provided)

        Returns
        -------
        X : np.ndarray
            Dense matrix of shape ``(n_rows, f)`` (or smaller when inferred from
            ``ids`` / ``start`` / ``stop``).
        ids_out : np.ndarray | None
            The materialized ids (dtype ``int64``) if ``include_ids=True``,
            otherwise ``None``.
        """
        self._require_numpy()

        n = self._n_rows(ids, start=start, stop=stop, n_rows=n_rows)
        if n < 0:
            raise ValueError("`n_rows` must be non-negative.")

        it, first_id, first_vec = self._iter_ids_with_first_vector(
            ids, start=start, stop=stop
        )
        f = self._infer_f_from_first_vec(first_vec)

        # Handle the empty selection early.
        if n == 0:
            # Strict mismatch detection for non-sized iterables:
            # `_iter_ids_with_first_vector` already consumes one id (if any).
            if first_id is not None:
                raise ValueError("`n_rows=0` but `ids` yielded at least one id.")
            X0 = np.empty((0, f), dtype=dtype)  # noqa: N806
            ids0 = np.empty((0,), dtype=np.int64) if include_ids else None
            return X0, ids0

        if first_id is None or first_vec is None:
            # n>0 but iterator is empty => mismatch between n_rows/ids range
            raise ValueError(
                "No ids available to materialize (empty selection), but `n_rows` is positive."
            )

        X = np.empty((n, f), dtype=dtype)
        ids_out = np.empty((n,), dtype=np.int64) if include_ids else None

        # Write first row
        v0 = np.asarray(first_vec, dtype=dtype)
        if validate_vector_len and v0.shape != (f,):
            raise ValueError(
                f"Vector length mismatch for id {first_id}: expected {f}, got {v0.shape}."
            )
        X[0] = v0
        if ids_out is not None:
            ids_out[0] = int(first_id)

        # Write remaining rows
        r = 1
        for item_id in it:
            if r >= n:
                # Strict: do not implicitly truncate when n_rows is provided.
                raise ValueError(
                    "`ids` yielded more items than expected `n_rows`/selection."
                )
            vec = np.asarray(self.get_item_vector(int(item_id)), dtype=dtype)
            if validate_vector_len and vec.shape != (f,):
                raise ValueError(
                    f"Vector length mismatch for id {int(item_id)}: expected {f}, got {vec.shape}."
                )
            X[r] = vec
            if ids_out is not None:
                ids_out[r] = int(item_id)
            r += 1

        if r != n:
            raise ValueError(
                "`ids` yielded fewer items than expected `n_rows`/selection."
            )

        return X, ids_out

    # ------------------------------------------------------------------ #
    # Public: streaming iteration
    # ------------------------------------------------------------------ #
    def iter_item_vectors(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        with_ids: bool = True,
    ) -> Iterator[Sequence[float] | tuple[int, Sequence[float]]]:
        """
        Iterate item vectors (memory-safe).

        Parameters
        ----------
        ids : sequence[int] or iterable[int], optional
            Explicit ids to iterate. If omitted, iterates over range(start, stop).
        start, stop : int, optional
            Used only when ids is None.
        with_ids : bool, default=True
            If True yield (id, vector); else yield vector.

        Yields
        ------
        (id, vector) or vector

        Notes
        -----
        This method does not allocate a matrix and is safe for large exports.
        """
        for item_id in self._iter_ids(ids, start=start, stop=stop):
            vec = self.get_item_vector(int(item_id))
            yield (int(item_id), vec) if with_ids else vec

    # ------------------------------------------------------------------ #
    # Public: NumPy exports
    # ------------------------------------------------------------------ #
    def to_numpy(
        self,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",  # type: ignore[]
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        validate_vector_len: bool = True,
    ) -> np.ndarray:  # type: ignore[]
        """
        Export vectors to a dense NumPy array.

        Parameters
        ----------
        ids : sequence of int or iterable of int, optional
            Item ids to export. If ``None``, exports the range
            ``[start, stop)`` (with ``stop`` defaulting to ``get_n_items()``).
            If you pass a non-sized iterable (e.g., a generator), you **must**
            pass ``n_rows`` and the iterable must yield exactly ``n_rows`` ids.
        dtype : str or numpy.dtype, default='float32'
            Output dtype.
        start, stop : int, optional
            Slice bounds used when ``ids`` is None.
        n_rows : int, optional
            Required when ``ids`` is a non-sized iterable. This function is
            strict: if the iterable yields fewer or more ids than ``n_rows``,
            a ``ValueError`` is raised.
        validate_vector_len : bool, default=True
            If True, verify that every vector has length ``f``. Disable only if
            you know your index can return ragged vectors (uncommon).

        Returns
        -------
        X : numpy.ndarray, shape (n_rows, f)
            Dense matrix of exported vectors.

        Notes
        -----
        - This export is deterministic: row order is defined by ``ids`` (or the
          ``start/stop`` range).
        - This export is single-pass and supports one-shot iterables.

        See Also
        --------
        iter_item_vectors : Stream vectors without materializing a dense matrix.
        to_dataframe : Export to a pandas DataFrame.
        save_vectors_npy : Persist vectors to an ``.npy`` memmap-able file.
        save_vectors_npz : Persist vectors to a compressed ``.npz`` archive.
        """
        X, _ = self._materialize_dense(
            ids,
            dtype=dtype,
            start=start,
            stop=stop,
            n_rows=n_rows,
            validate_vector_len=validate_vector_len,
            include_ids=False,
        )
        return X

    def save_vectors_npy(  # noqa: PLR0912
        self,
        path: PathLikeStr,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",  # type: ignore[]
        start: int = 0,
        stop: int | None = None,
        overwrite: bool = True,
        n_rows: int | None = None,
        flush_every: int | None = None,
        validate_vector_len: bool = True,
    ) -> str:
        """
        Write vectors to an ``.npy`` file using a memory-mapped array.

        This is the recommended export format for large indices because the
        resulting file can be memory-mapped with ``np.load(..., mmap_mode='r')``
        without reading the entire array into RAM.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path. The ``.npy`` extension is recommended.
        ids : sequence of int or iterable of int, optional
            Item ids to export. If ``None``, exports the range ``[start, stop)``.
            If you pass a non-sized iterable, you **must** pass ``n_rows`` and
            the iterable must yield exactly ``n_rows`` ids.
        dtype : str or numpy.dtype, default='float32'
            Stored dtype.
        start, stop : int, optional
            Slice bounds used when ``ids`` is None.
        overwrite : bool, default=True
            If False and ``path`` exists, raise ``FileExistsError``.
        n_rows : int, optional
            Required when ``ids`` is a non-sized iterable.
        flush_every : int, optional
            Flush the memmap to disk after every N rows (useful for very large
            exports). If None, flush once at the end.
        validate_vector_len : bool, default=True
            If True, verify that every vector has length ``f``.

        Returns
        -------
        out_path : str
            Absolute path of the written file.

        Notes
        -----
        This function is strict and deterministic:

        - No implicit truncation when ``n_rows`` is provided.
        - Output row order follows the provided ``ids`` / range selection.

        See Also
        --------
        save_vectors_npz : Small/portable ``.npz`` export.
        to_numpy : Materialize the full matrix in memory.
        """
        self._require_numpy()

        out = self._fspath(path)
        if (not overwrite) and os.path.exists(out):
            raise FileExistsError(out)

        n = self._n_rows(ids, start=start, stop=stop, n_rows=n_rows)
        if n < 0:
            raise ValueError("`n_rows` must be non-negative.")

        it, first_id, first_vec = self._iter_ids_with_first_vector(
            ids, start=start, stop=stop
        )
        f = self._infer_f_from_first_vec(first_vec)

        if n == 0:
            # Strict mismatch detection for non-sized iterables:
            # `_iter_ids_with_first_vector` already consumes one id (if any).
            if first_id is not None:
                raise ValueError("`n_rows=0` but `ids` yielded at least one id.")
            arr = open_memmap(out, mode="w+", dtype=dtype, shape=(0, f))
            arr.flush()
            return os.path.abspath(out)

        if first_id is None or first_vec is None:
            raise ValueError("No ids available to export, but `n_rows` is positive.")

        arr = open_memmap(out, mode="w+", dtype=dtype, shape=(n, f))

        v0 = np.asarray(first_vec, dtype=dtype)
        if validate_vector_len and v0.shape != (f,):
            raise ValueError(
                f"Vector length mismatch for id {first_id}: expected {f}, got {v0.shape}."
            )
        arr[0] = v0

        r = 1
        for item_id in it:
            if r >= n:
                raise ValueError(
                    "`ids` yielded more items than expected `n_rows`/selection."
                )
            vec = np.asarray(self.get_item_vector(int(item_id)), dtype=dtype)
            if validate_vector_len and vec.shape != (f,):
                raise ValueError(
                    f"Vector length mismatch for id {int(item_id)}: expected {f}, got {vec.shape}."
                )
            arr[r] = vec
            r += 1

            if (
                flush_every is not None
                and flush_every > 0
                and (r % int(flush_every) == 0)
            ):
                arr.flush()

        if r != n:
            raise ValueError(
                "`ids` yielded fewer items than expected `n_rows`/selection."
            )

        arr.flush()
        return os.path.abspath(out)

    def save_vectors_npz(
        self,
        path: PathLikeStr,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",  # type: ignore[]
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        compressed: bool = True,
        include_ids: bool = True,
        array_name: str = "vectors",
        ids_name: str = "ids",
        validate_vector_len: bool = True,
    ) -> str:
        """
        Write vectors to a NumPy ``.npz`` archive.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path. The ``.npz`` extension is recommended.
        ids : sequence of int or iterable of int, optional
            Item ids to export. If ``None``, exports the range ``[start, stop)``.
            If you pass a non-sized iterable, you **must** pass ``n_rows`` and
            the iterable must yield exactly ``n_rows`` ids.
        dtype : str or numpy.dtype, default='float32'
            Stored dtype.
        start, stop : int, optional
            Slice bounds used when ``ids`` is None.
        n_rows : int, optional
            Required when ``ids`` is a non-sized iterable.
        compressed : bool, default=True
            If True, uses ``np.savez_compressed``.
        include_ids : bool, default=True
            If True, store an ``ids`` array alongside the vectors.
        array_name : str, default='vectors'
            Key used for the vectors array inside the archive.
        ids_name : str, default='ids'
            Key used for the ids array inside the archive (when ``include_ids=True``).
        validate_vector_len : bool, default=True
            If True, verify that every vector has length ``f``.

        Returns
        -------
        out_path : str
            Absolute path of the written file.

        Notes
        -----
        ``.npz`` is convenient for portability, but it requires loading the full
        array into memory during export. For very large indices, prefer
        :meth:`save_vectors_npy`.

        See Also
        --------
        save_vectors_npy : Large/streaming-friendly ``.npy`` export.
        to_numpy : In-memory export.
        """
        self._require_numpy()

        X, ids_out = self._materialize_dense(
            ids,
            dtype=dtype,
            start=start,
            stop=stop,
            n_rows=n_rows,
            validate_vector_len=validate_vector_len,
            include_ids=include_ids,
        )

        out = os.path.abspath(self._fspath(path))
        saver = np.savez_compressed if compressed else np.savez

        if include_ids:
            assert ids_out is not None  # noqa: S101
            saver(out, **{array_name: X, ids_name: ids_out})
        else:
            saver(out, **{array_name: X})

        return out

    # ------------------------------------------------------------------ #
    # Public: pandas / CSV
    # ------------------------------------------------------------------ #
    def to_dataframe(
        self,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",  # type: ignore[]
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        include_id: bool = True,
        columns: Sequence[str] | None = None,
        validate_vector_len: bool = True,
    ):
        """
        Export vectors to a pandas ``DataFrame``.

        Parameters
        ----------
        ids : sequence of int or iterable of int, optional
            Item ids to export. If ``None``, exports the range ``[start, stop)``.
            If you pass a non-sized iterable, you **must** pass ``n_rows`` and
            the iterable must yield exactly ``n_rows`` ids.
        dtype : str or numpy.dtype, default='float32'
            Stored dtype.
        start, stop : int, optional
            Slice bounds used when ``ids`` is None.
        n_rows : int, optional
            Required when ``ids`` is a non-sized iterable.
        include_id : bool, default=True
            If True, include an integer ``id`` column as the first column.
        columns : sequence of str, optional
            Column names for vector dimensions. If None, uses ``feature_0..feature_{f-1}``.
        validate_vector_len : bool, default=True
            If True, verify that every vector has length ``f``.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with shape ``(n_rows, f + include_id)``.

        Notes
        -----
        This function materializes a dense matrix in memory. For very large
        indices, consider :meth:`to_parquet`, :meth:`to_sqlite`, or
        :meth:`save_vectors_npy`.

        See Also
        --------
        to_numpy : In-memory dense export.
        to_csv : Streaming text export.
        to_parquet : Columnar export for analytics stacks.
        """
        self._require_pandas()
        self._require_numpy()

        X, ids_out = self._materialize_dense(
            ids,
            dtype=dtype,
            start=start,
            stop=stop,
            n_rows=n_rows,
            validate_vector_len=validate_vector_len,
            include_ids=include_id,
        )
        f = int(X.shape[1])

        if columns is None:
            cols = [f"feature_{j}" for j in range(f)]
        else:
            cols = list(columns)
            if len(cols) != f:
                raise ValueError(f"`columns` must have length {f}, got {len(cols)}.")

        df = pd.DataFrame(X, columns=cols)

        if include_id:
            assert ids_out is not None  # noqa: S101
            df.insert(0, "id", ids_out.astype("int64", copy=False))
        return df

    def to_csv(
        self,
        path: PathLikeStr,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",  # type: ignore[]
        start: int = 0,
        stop: int | None = None,
        include_id: bool = True,
        header: bool = True,
        delimiter: str = ",",
        float_format: str | None = None,
        columns: Sequence[str] | None = None,
        encoding: str = "utf-8",
        validate_vector_len: bool = True,
    ) -> str:
        r"""
        Export vectors to a delimited text file (CSV/TSV).

        This is a streaming export: vectors are read and written row-by-row.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path.
        ids : sequence of int or iterable of int, optional
            Item ids to export. If ``None``, exports the range ``[start, stop)``.
        dtype : str or numpy.dtype, default='float32'
            Conversion dtype for values written to disk.
        start, stop : int, optional
            Slice bounds used when ``ids`` is None.
        include_id : bool, default=True
            If True, include an integer ``id`` column.
        header : bool, default=True
            If True, write a header row.
        delimiter : str, default=','
            Column delimiter (use ``'\t'`` for TSV).
        float_format : str, optional
            Format string for floats (e.g. ``'%.6g'``). If None, uses default
            string conversion.
        columns : sequence of str, optional
            Column names for vector dimensions. If None, uses ``feature_0..feature_{f-1}``.
        encoding : str, default='utf-8'
            Output file encoding.
        validate_vector_len : bool, default=True
            If True, verify that every vector has length ``f``.

        Returns
        -------
        out_path : str
            Absolute path of the written file.

        See Also
        --------
        to_parquet : Efficient columnar export.
        to_sqlite : Export to a lightweight relational database.
        """
        self._require_numpy()

        out = os.path.abspath(self._fspath(path))
        it, first_id, first_vec = self._iter_ids_with_first_vector(
            ids, start=start, stop=stop
        )
        f = self._infer_f_from_first_vec(first_vec)

        if columns is None:
            cols = [f"feature_{j}" for j in range(f)]
        else:
            cols = list(columns)
            if len(cols) != f:
                raise ValueError(f"`columns` must have length {f}, got {len(cols)}.")

        fmt = float_format

        def _format_row(row):
            if fmt is None:
                return row
            return [fmt % float(v) for v in row]

        with open(out, "w", newline="", encoding=encoding) as fp:
            writer = csv.writer(fp, delimiter=delimiter)
            if header:
                header_cols = (["id"] if include_id else []) + cols
                writer.writerow(header_cols)

            if first_id is not None and first_vec is not None:
                v0 = np.asarray(first_vec, dtype=dtype)
                if validate_vector_len and v0.shape != (f,):
                    raise ValueError(
                        f"Vector length mismatch for id {first_id}: expected {f}, got {v0.shape}."
                    )
                row = v0.tolist()
                row = _format_row(row)
                if include_id:
                    writer.writerow([int(first_id), *row])
                else:
                    writer.writerow(row)

            for item_id in it:
                vec = np.asarray(self.get_item_vector(int(item_id)), dtype=dtype)
                if validate_vector_len and vec.shape != (f,):
                    raise ValueError(
                        f"Vector length mismatch for id {int(item_id)}: expected {f}, got {vec.shape}."
                    )
                row = vec.tolist()
                row = _format_row(row)
                if include_id:
                    writer.writerow([int(item_id), *row])
                else:
                    writer.writerow(row)

        return out

    # ------------------------------------------------------------------ #
    # Public: Parquet (pyarrow), SQLite, SciPy, MLflow
    # ------------------------------------------------------------------ #
    def to_parquet(  # noqa: PLR0912
        self,
        path: PathLikeStr,
        ids: IdsInput = None,
        *,
        dtype: str | np.dtype = "float32",  # type: ignore[]
        start: int = 0,
        stop: int | None = None,
        include_id: bool = True,
        columns: Sequence[str] | None = None,
        batch_size: int = 50_000,
        compression: str | None = "snappy",
        validate_vector_len: bool = True,
    ) -> str:
        """
        Export vectors to an Apache Parquet file.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file path.
        ids : sequence of int or iterable of int, optional
            Item ids to export. If ``None``, exports the range ``[start, stop)``.
        dtype : str or numpy.dtype, default='float32'
            Stored dtype for vector columns.
        start, stop : int, optional
            Slice bounds used when ``ids`` is None.
        include_id : bool, default=True
            If True, include an integer ``id`` column.
        columns : sequence of str, optional
            Column names for vector dimensions. If None, uses ``feature_0..feature_{f-1}``.
        batch_size : int, default=50000
            Number of rows per Parquet write batch.
        compression : str or None, default='snappy'
            Compression codec (requires Arrow build support).
        validate_vector_len : bool, default=True
            If True, verify that every vector has length ``f``.

        Returns
        -------
        out_path : str
            Absolute path of the written file.

        Raises
        ------
        ImportError
            If ``pyarrow`` is not available.

        See Also
        --------
        to_csv : Human-readable streaming export.
        to_sqlite : Lightweight relational export.
        """
        self._require_numpy()

        if int(batch_size) <= 0:
            raise ValueError("`batch_size` must be a positive integer.")

        try:
            import pyarrow as pa  # noqa: PLC0415
            import pyarrow.parquet as pq  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError("pyarrow is required for Parquet export") from e

        out = os.path.abspath(self._fspath(path))

        it, first_id, first_vec = self._iter_ids_with_first_vector(
            ids, start=start, stop=stop
        )
        f = self._infer_f_from_first_vec(first_vec)

        if columns is None:
            cols = [f"feature_{j}" for j in range(f)]
        else:
            cols = list(columns)
            if len(cols) != f:
                raise ValueError(f"`columns` must have length {f}, got {len(cols)}.")

        # Stable schema (no inference drift).
        fields = []
        if include_id:
            fields.append(pa.field("id", pa.int64()))
        arrow_dtype = pa.from_numpy_dtype(np.dtype(dtype))
        fields.extend(pa.field(c, arrow_dtype) for c in cols)
        schema = pa.schema(fields)

        writer = pq.ParquetWriter(out, schema=schema, compression=compression)

        def _flush(batch_ids, batch_rows):
            arrays = []
            if include_id:
                arrays.append(pa.array(batch_ids, type=pa.int64()))
            # transpose rows -> columns
            cols_data = list(zip(*batch_rows)) if batch_rows else [[] for _ in range(f)]
            for col in cols_data:
                arrays.append(pa.array(col, type=arrow_dtype))
            table = pa.Table.from_arrays(arrays, schema=schema)
            writer.write_table(table)

        batch_ids = []
        batch_rows = []

        try:
            # Include the first row if present.
            if first_id is not None and first_vec is not None:
                v0 = np.asarray(first_vec, dtype=dtype)
                if validate_vector_len and v0.shape != (f,):
                    raise ValueError(
                        f"Vector length mismatch for id {first_id}: expected {f}, got {v0.shape}."
                    )
                if include_id:
                    batch_ids.append(int(first_id))
                batch_rows.append(v0.tolist())

            for item_id in it:
                vec = np.asarray(self.get_item_vector(int(item_id)), dtype=dtype)
                if validate_vector_len and vec.shape != (f,):
                    raise ValueError(
                        f"Vector length mismatch for id {int(item_id)}: expected {f}, got {vec.shape}."
                    )
                if include_id:
                    batch_ids.append(int(item_id))
                batch_rows.append(vec.tolist())

                if len(batch_rows) >= int(batch_size):
                    _flush(batch_ids, batch_rows)
                    batch_ids = []
                    batch_rows = []

            if batch_rows:
                _flush(batch_ids, batch_rows)
        finally:
            writer.close()

        return out

    def to_sqlite(  # noqa: PLR0912
        self,
        path: PathLikeStr,
        *,
        table: str = "annoy_vectors",
        ids: IdsInput = None,
        dtype: str | np.dtype = "float32",  # type: ignore[]
        start: int = 0,
        stop: int | None = None,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        vector_format: Literal["blob", "json"] = "blob",
        batch_size: int = 10_000,
        validate_vector_len: bool = True,
    ) -> str:
        """
        Export vectors to a SQLite database table.

        This export supports two storage formats:

        - ``vector_format='blob'``: store a raw NumPy byte string (compact & fast)
        - ``vector_format='json'``: store a JSON array (human-readable, larger)

        Parameters
        ----------
        path : str or os.PathLike
            SQLite database file path.
        table : str, default='annoy_vectors'
            Destination table name. Must be a valid SQLite identifier
            (letters/digits/underscore, not quoted).
        ids : sequence of int or iterable of int, optional
            Item ids to export. If ``None``, exports the range ``[start, stop)``.
        dtype : str or numpy.dtype, default='float32'
            dtype used when encoding vectors.
        start, stop : int, optional
            Slice bounds used when ``ids`` is None.
        if_exists : {'fail', 'replace', 'append'}, default='fail'
            Behavior if the table already exists.
        vector_format : {'blob', 'json'}, default='blob'
            Storage format for vectors.
        batch_size : int, default=10000
            Number of rows per database commit.
        validate_vector_len : bool, default=True
            If True, verify that every vector has length ``f``.

        Returns
        -------
        out_path : str
            Absolute path of the SQLite database file.

        Notes
        -----
        - Inserts use ``INSERT OR REPLACE`` so exporting the same id multiple
          times overwrites previous rows deterministically.
        - This function does not attempt schema migration; changing ``dtype`` or
          ``vector_format`` requires ``if_exists='replace'`` or a new table name.

        See Also
        --------
        to_parquet : Columnar export for analytics tools.
        to_csv : Text export for lightweight interoperability.
        """
        self._require_numpy()

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
            raise ValueError(
                "`table` must be a valid SQLite identifier (letters/digits/underscore)."
            )

        if int(batch_size) <= 0:
            raise ValueError("`batch_size` must be a positive integer.")

        if vector_format not in ("blob", "json"):
            raise ValueError("`vector_format` must be either 'blob' or 'json'.")

        out = os.path.abspath(self._fspath(path))
        con = sqlite3.connect(out)
        try:
            cur = con.cursor()

            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (table,),
            )
            exists = cur.fetchone() is not None

            if exists:
                if if_exists == "fail":
                    raise ValueError(f"Table {table!r} already exists.")
                if if_exists == "replace":
                    cur.execute(f'DROP TABLE IF EXISTS "{table}";')
            else:  # noqa: PLR5501
                if if_exists not in ("fail", "replace", "append"):
                    raise ValueError(f"Unknown if_exists={if_exists!r}.")

            # Determine dimension and create schema.
            it, first_id, first_vec = self._iter_ids_with_first_vector(
                ids, start=start, stop=stop
            )
            f = self._infer_f_from_first_vec(first_vec)

            cur.execute(
                f'CREATE TABLE IF NOT EXISTS "{table}" ('
                "id INTEGER PRIMARY KEY, "
                "vector BLOB, "
                "f INTEGER NOT NULL, "
                "dtype TEXT NOT NULL, "
                "vector_format TEXT NOT NULL"
                ");"
            )

            def encode(vec):
                v = np.asarray(vec, dtype=dtype)
                if validate_vector_len and v.shape != (f,):
                    raise ValueError(
                        f"Vector length mismatch: expected {f}, got {v.shape}."
                    )
                if vector_format == "blob":
                    return v.tobytes(order="C")
                return json.dumps(v.tolist(), separators=(",", ":")).encode("utf-8")

            batch = []

            def flush():
                if not batch:
                    return
                cur.executemany(
                    f'INSERT OR REPLACE INTO "{table}" (id, vector, f, dtype, vector_format) VALUES (?, ?, ?, ?, ?);',  # noqa: S608
                    batch,
                )
                con.commit()
                batch.clear()

            if first_id is not None and first_vec is not None:
                batch.append(
                    (
                        int(first_id),
                        encode(first_vec),
                        int(f),
                        str(np.dtype(dtype)),
                        vector_format,
                    )
                )
                if len(batch) >= int(batch_size):
                    flush()

            for item_id in it:
                batch.append(
                    (
                        int(item_id),
                        encode(self.get_item_vector(int(item_id))),
                        int(f),
                        str(np.dtype(dtype)),
                        vector_format,
                    )
                )
                if len(batch) >= int(batch_size):
                    flush()

            flush()
        finally:
            con.close()

        return out

    def to_scipy_csr(
        self,
        ids: IdsInput = None,
        *,
        dtype: str | Any = "float32",
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
    ):
        """
        Convert vectors to a SciPy CSR sparse matrix (requires SciPy + NumPy).

        Parameters
        ----------
        ids : sequence of int or iterable of int, optional
            Item ids to export. If ``None``, exports the range ``[start, stop)``.
            If you pass a non-sized iterable (e.g., a generator), you **must**
            pass ``n_rows`` and the iterable must yield exactly ``n_rows`` ids.
        dtype : str or numpy.dtype, default='float32'
            dtype used when building the dense intermediate matrix.
        start, stop : int, optional
            Slice bounds used when ``ids`` is None.
        n_rows : int, optional
            Required when ``ids`` is a non-sized iterable.

        Returns
        -------
        X : scipy.sparse.csr_matrix
            CSR matrix with shape ``(n_rows, f)``.

        Raises
        ------
        RuntimeError
            If SciPy is not installed.

        Notes
        -----
        This is a convenience wrapper equivalent to::

            scipy.sparse.csr_matrix(self.to_numpy(...))

        See Also
        --------
        to_numpy : Materialize vectors to a dense NumPy array.
        """
        self._require_numpy()
        try:  # pragma: no cover
            import scipy.sparse as sp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("SciPy is required for sparse export.") from e

        X = self.to_numpy(ids, dtype=dtype, start=start, stop=stop, n_rows=n_rows)
        return sp.csr_matrix(X)

    def log_export_to_mlflow(
        self,
        artifact_path: PathLikeStr,
        *,
        mlflow_artifact_path: str = "annoy_vectors",
        log_params: bool = True,
        extra_params: Mapping[str, Any] | None = None,
    ) -> str:
        """
        Log an already-created export file to MLflow as an artifact.

        Parameters
        ----------
        artifact_path : str or path-like
            File path created by this module (csv/parquet/npy/npz/sqlite/etc.).
        mlflow_artifact_path : str, default='annoy_vectors'
            Artifact subdirectory inside the run.
        log_params : bool, default=True
            If True, logs stable metadata params when available.
        extra_params : mapping, optional
            Additional params to log (converted deterministically to strings).

        Returns
        -------
        artifact_path : str

        Notes
        -----
        This method does not create or manage MLflow runs. An active run must
        exist (or MLflow must be configured externally).
        """
        out = os.path.abspath(self._fspath(artifact_path))
        if not os.path.exists(out):
            raise FileNotFoundError(out)
        try:  # pragma: no cover
            import mlflow  # type: ignore[] # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("mlflow is required for MLflow logging.") from e

        mlflow.log_artifact(out, artifact_path=mlflow_artifact_path)

        if log_params:
            params: dict[str, Any] = {}
            params["annoy_f"] = int(getattr(self, "f", 0) or 0)
            if hasattr(self, "metric"):
                params["annoy_metric"] = getattr(self, "metric", None)
            if hasattr(self, "get_n_items"):
                params["annoy_n_items"] = int(self.get_n_items())
            if hasattr(self, "get_n_trees"):
                try:  # noqa: SIM105
                    params["annoy_n_trees"] = int(self.get_n_trees())  # type: ignore[attr-defined]
                except Exception:
                    pass
            if extra_params is not None:
                params.update(dict(extra_params))

            mlflow.log_params({k: str(v) for k, v in params.items()})

        return out

    # ------------------------------------------------------------------ #
    # Public: ID existence partition (explicit, non-assuming)
    # ------------------------------------------------------------------ #
    def partition_existing_ids(
        self,
        ids: Sequence[int],
        missing_exceptions: tuple[type[BaseException], ...] = (IndexError,),
    ) -> tuple[list[int], list[int]]:
        """
        Partition candidate ids into (existing, missing) by calling get_item_vector.

        Parameters
        ----------
        ids : sequence[int]
            Candidate ids.
        missing_exceptions : tuple[type[Exception], ...], default=(IndexError,)
            Exceptions treated as "missing". Anything else is re-raised.

        Returns
        -------
        existing : list[int]
        missing : list[int]

        Notes
        -----
        This method intentionally does not assume ids are contiguous.
        """
        existing: list[int] = []
        missing: list[int] = []

        for i in ids:
            ii = int(i)
            try:
                _ = self.get_item_vector(ii)
            except missing_exceptions:
                missing.append(ii)
            except Exception:
                raise
            else:
                existing.append(ii)

        return existing, missing
