# _ndarray.py


"""
NDArrayExportMixin for :py:mod:`~scikitplot.cexternals.annoy` .

High-level export utilities for Annoy Index objects.

Design goals:
- Strict, deterministic API.
- No C-API changes.
- Memory-safe options for very large indexes (e.g., 1B items).
- Optional NumPy/Pandas dependencies.
"""

from __future__ import annotations

import csv
from typing import (
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Union,
)

try:
    import numpy as np
    from numpy.lib.format import open_memmap
except Exception:  # pragma: no cover
    np = None
    open_memmap = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


IdsInput = Optional[Union[Sequence[int], Iterable[int]]]


class NDArrayExportMixin:
    """
    Export mixin for Annoy-like classes.

    A class mixing this in MUST provide:
    - get_item_vector(i: int) -> Sequence[float]
    - get_n_items() -> int
    - attribute/property: f (dimension)
    """

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _require_numpy(self) -> None:
        if np is None or open_memmap is None:
            raise RuntimeError("NumPy is required for this operation")

    def _require_pandas(self) -> None:
        if pd is None:
            raise RuntimeError("Pandas is required for this operation")

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

    # -----------------------------
    # Public iteration
    # -----------------------------
    def iter_item_vectors(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        with_ids: bool = True,
    ) -> Iterator[Sequence[float] | tuple[int, Sequence[float]]]:
        """
        Iterate item vectors in a memory-safe way.

        Parameters
        ----------
        ids:
            Explicit item ids. Must be a sized Sequence for strictness.
        start, stop:
            Used only when ids is None.
        with_ids:
            If True yield (id, vector), else yield vector.

        Yields
        ------
        (id, vector) or vector
        """
        iterable_ids, _ = self._normalize_ids(ids, start=start, stop=stop)
        for i in iterable_ids:
            ii = int(i)
            vec = self.get_item_vector(ii)
            yield (ii, vec) if with_ids else vec

    # -----------------------------
    # In-memory matrix (small/medium)
    # -----------------------------
    def to_numpy(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        dtype: str = "float32",
    ):
        """
        Materialize vectors into an in-memory NumPy array.

        WARNING: Not suitable for huge indexes.
        """
        self._require_numpy()
        iterable_ids, n_rows = self._normalize_ids(ids, start=start, stop=stop)

        f = int(getattr(self, "f", 0))
        if f <= 0:
            # infer dimension from first vector
            first_id = None
            if isinstance(iterable_ids, range):
                if n_rows == 0:
                    return np.empty((0, 0), dtype=dtype)
                first_id = int(iterable_ids.start)
            elif isinstance(iterable_ids, Sequence) and len(iterable_ids) > 0:
                first_id = int(iterable_ids[0])
            if first_id is None:
                raise ValueError("Cannot infer dimension from empty ids")
            f = len(self.get_item_vector(first_id))

        arr = np.empty((n_rows, f), dtype=dtype)
        for row_idx, i in enumerate(iterable_ids):
            arr[row_idx, :] = np.asarray(self.get_item_vector(int(i)), dtype=dtype)

        return arr

    # -----------------------------
    # On-disk .npy via memmap (large)
    # -----------------------------
    def save_vectors_npy(
        self,
        path: str,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        dtype: str = "float32",
        overwrite: bool = True,
    ) -> str:
        """
        Save vectors into a .npy file using NumPy open_memmap.

        This is the recommended path for very large indexes.

        Returns
        -------
        path
        """
        self._require_numpy()

        iterable_ids, n_rows = self._normalize_ids(ids, start=start, stop=stop)

        f = int(getattr(self, "f", 0))
        if f <= 0:
            # infer if needed
            first_id = None
            if isinstance(iterable_ids, range):
                if n_rows == 0:
                    f = 0
                else:
                    first_id = int(iterable_ids.start)
            elif isinstance(iterable_ids, Sequence) and len(iterable_ids) > 0:
                first_id = int(iterable_ids[0])
            if first_id is not None:
                f = len(self.get_item_vector(first_id))

        if f <= 0 and n_rows > 0:
            raise ValueError(
                "Vector dimension (f) is not set and could not be inferred"
            )

        mode = "w+" if overwrite else "w+"  # noqa: RUF034

        mm = open_memmap(
            path,
            mode=mode,
            dtype=dtype,
            shape=(n_rows, f),
        )

        for row_idx, i in enumerate(iterable_ids):
            mm[row_idx, :] = np.asarray(self.get_item_vector(int(i)), dtype=dtype)

        # ensure flush
        del mm
        return path

    # -----------------------------
    # Pandas DataFrame (small/medium)
    # -----------------------------
    def to_dataframe(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        include_id: bool = True,
        columns: list[str] | None = None,
        dtype: str = "float32",
    ):
        """
        Materialize vectors into a Pandas DataFrame.

        WARNING: Not suitable for huge indexes.
        """
        self._require_pandas()
        self._require_numpy()

        arr = self.to_numpy(ids, start=start, stop=stop, dtype=dtype)
        n_rows, f = arr.shape  # noqa: RUF059

        if columns is None:
            columns = [f"feature_{k}" for k in range(f)]
        elif len(columns) != f:
            raise ValueError("columns length must match vector dimension")

        df = pd.DataFrame(arr, columns=columns)
        if include_id:
            iterable_ids, _ = self._normalize_ids(ids, start=start, stop=stop)
            df.insert(0, "id", [int(i) for i in iterable_ids])

        return df

    # -----------------------------
    # Streaming CSV writer (large)
    # -----------------------------
    def to_csv(  # noqa: PLR0912
        self,
        path: str,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        include_id: bool = True,
        header: bool = True,
        delimiter: str = ",",
        float_format: str | None = None,
        columns: list[str] | None = None,
        dtype: str = "float32",
    ) -> str:
        """
        Stream vectors to CSV without building a full DataFrame.

        This is safer than df.to_csv for large exports.

        Notes
        -----
        CSV for 1B rows will be extremely large and slow.
        Consider Parquet in the future.
        """
        self._require_numpy()

        iterable_ids, n_rows = self._normalize_ids(ids, start=start, stop=stop)

        f = int(getattr(self, "f", 0))
        if f <= 0 and n_rows > 0:
            # infer from first id (strict)
            first_id = None
            if isinstance(iterable_ids, range):
                first_id = int(iterable_ids.start)
            elif isinstance(iterable_ids, Sequence) and len(iterable_ids) > 0:
                first_id = int(iterable_ids[0])
            if first_id is not None:
                f = len(self.get_item_vector(first_id))

        if columns is None:
            columns = [f"feature_{k}" for k in range(f)]
        elif len(columns) != f:
            raise ValueError("columns length must match vector dimension")

        # Recompute iterable_ids because we may have consumed assumptions
        iterable_ids, _ = self._normalize_ids(ids, start=start, stop=stop)

        with open(path, "w", newline="", encoding="utf-8") as fobj:
            writer = csv.writer(fobj, delimiter=delimiter)

            if header:
                if include_id:
                    writer.writerow(["id", *columns])
                else:
                    writer.writerow(columns)

            for i in iterable_ids:
                ii = int(i)
                vec = self.get_item_vector(ii)

                # strict length check
                if len(vec) != len(columns):
                    raise ValueError(
                        f"Vector length mismatch for id={ii}: "
                        f"{len(vec)} != {len(columns)}"
                    )

                if float_format:
                    row_vals = [format(float(x), float_format) for x in vec]
                else:
                    row_vals = [float(x) for x in vec]

                if include_id:
                    writer.writerow([ii, *row_vals])
                else:
                    writer.writerow(row_vals)

        return path
