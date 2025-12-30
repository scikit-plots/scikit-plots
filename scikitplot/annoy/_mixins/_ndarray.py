# scikitplot/annoy/_mixins/_ndarray.py
"""
NumPy / SciPy / pandas interoperability.

Defines :class:`~scikitplot.annoy._mixins._ndarray.NDArrayMixin`, a **NumPy-first**,
**explicit**, **deterministic** mixin providing batch add and export utilities
for Annoy-like indexes.

The mixin expects that ``backend_for(self)`` returns an object implementing:

- ``add_item(i: int, vector: Sequence[float])``
- ``get_item_vector(i: int) -> Sequence[float]``
- ``get_n_items() -> int``
- Optional: ``get_n_trees() -> int`` (built-state detection)
- Optional: attribute/property ``f`` (dimension)

See Also
--------
scikitplot.annoy._mixins._vectors.VectorOpsMixin
    Query-side neighbor utilities (kNN).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np

from .._utils import backend_for, lock_for

__all__: tuple[str, ...] = ("NDArrayMixin",)

IdsInput: TypeAlias = Sequence[int] | Iterable[int] | None
SparsePolicy: TypeAlias = Literal["error", "toarray"]
FinitePolicy: TypeAlias = bool | Literal["allow-nan"]


# ---------------------------------------------------------------------------
# Optional dependencies (lazy)
# ---------------------------------------------------------------------------


def _lazy_import_scipy_sparse() -> Any:
    """
    Import SciPy sparse lazily.

    Raises
    ------
    ImportError
        If SciPy is not installed.
    """
    try:
        import scipy.sparse as sp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise ImportError("SciPy is required for this operation.") from e
    return sp


def _lazy_import_pandas() -> Any:
    """
    Import pandas lazily.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise ImportError("pandas is required for this operation.") from e
    return pd


# ---------------------------------------------------------------------------
# Deterministic validation helpers
# ---------------------------------------------------------------------------


def _ensure_all_finite(X: np.ndarray, policy: FinitePolicy) -> None:
    """
    Deterministically validate finiteness.

    Parameters
    ----------
    X : numpy.ndarray
        Array to validate.
    policy : bool or {'allow-nan'}
        - True: no NaN and no inf
        - 'allow-nan': NaN allowed, inf forbidden
        - False: no finiteness checks

    Raises
    ------
    ValueError
        If the policy is violated.
    """
    if policy is False:
        return

    if policy is True:
        if not np.isfinite(X).all():
            raise ValueError(
                "X contains NaN or infinite values (ensure_all_finite=True)."
            )
        return

    # policy == "allow-nan"
    if np.isinf(X).any():
        raise ValueError("X contains infinite values (ensure_all_finite='allow-nan').")


def _as_dense_2d_float(
    X: Any,
    *,
    dtype: Any,
    order: Literal["C", "F", "A", "K"],
    copy: bool,
    accept_sparse: SparsePolicy,
    ensure_all_finite: FinitePolicy,
) -> np.ndarray:
    """
    Coerce input to a dense 2D float matrix deterministically.

    Parameters
    ----------
    X : array-like
        Input array-like. May be a NumPy array, pandas DataFrame/Series, or
        SciPy sparse matrix (depending on ``accept_sparse``).
    dtype : numpy dtype
        Output dtype (typically ``numpy.float32``).
    order : {'C', 'F', 'A', 'K'}
        Memory order passed to :func:`numpy.asarray`.
    copy : bool
        If True, force a copy.
    accept_sparse : {'error', 'toarray'}
        Sparse input handling.
    ensure_all_finite : bool or 'allow-nan'
        Finiteness policy.

    Returns
    -------
    Xv : numpy.ndarray of shape (n_samples, n_features)
        Dense float matrix.

    Raises
    ------
    TypeError
        If sparse input is provided while ``accept_sparse='error'``.
    ValueError
        If the input is not 2D or violates ``ensure_all_finite``.
    """
    # SciPy sparse handling (explicit, no guessing)
    sp = None
    try:
        sp = _lazy_import_scipy_sparse()
    except ImportError:
        sp = None

    if sp is not None and sp.issparse(X):
        if accept_sparse == "error":
            raise TypeError(
                "Sparse input is not accepted (accept_sparse='error'). "
                "Pass accept_sparse='toarray' to densify explicitly."
            )
        X = X.toarray()

    Xv = np.asarray(X, dtype=dtype, order=order)  # noqa: N806
    if Xv.ndim != 2:  # noqa: PLR2004
        raise ValueError(
            f"X must be 2D of shape (n_samples, n_features); got ndim={int(Xv.ndim)}."
        )

    if copy:
        Xv = Xv.copy(order=order)  # noqa: N806

    _ensure_all_finite(Xv, ensure_all_finite)
    return np.ascontiguousarray(Xv)


def _coerce_ids(ids: Any, *, n_samples: int) -> np.ndarray:
    """Coerce ids into an int64 1D array deterministically."""
    if ids is None:
        raise TypeError("ids is None; caller must generate ids deterministically.")

    if isinstance(ids, np.ndarray):
        ids_arr = ids.astype(np.int64, copy=False)
    else:
        try:
            ids_arr = np.asarray(list(ids), dtype=np.int64)
        except TypeError:
            raise TypeError("ids must be an iterable of integers.") from None

    if ids_arr.ndim != 1:
        raise ValueError("ids must be 1D.")
    if int(ids_arr.shape[0]) != int(n_samples):
        raise ValueError(
            f"ids length mismatch: expected {int(n_samples)}, got {int(ids_arr.shape[0])}."
        )
    if (ids_arr < 0).any():
        raise ValueError("ids must be non-negative.")
    return ids_arr


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class NDArrayMixin:
    """NumPy / SciPy / pandas interoperability for Annoy-like indexes."""

    # ------------------------------------------------------------------
    # Internal helpers (namespaced to reduce MRO collisions)
    # ------------------------------------------------------------------
    def _ndarray_require_unbuilt(self) -> None:
        """
        Raise if the backend indicates the index is built.

        Notes
        -----
        If the backend cannot reliably report build state, this method will not
        guess and will not raise.
        """
        backend = backend_for(self)
        get_n_trees = getattr(backend, "get_n_trees", None)
        if not callable(get_n_trees):
            return

        try:
            with lock_for(self):
                n_trees = int(get_n_trees())
        except Exception:
            # If the backend cannot report build state reliably, do not guess.
            return

        if n_trees > 0:
            raise RuntimeError(
                "Index is built; adding items is not supported. "
                "Call `unbuild()` (or create a new index) before adding items."
            )

    def _ndarray_iter_ids(
        self,
        ids: IdsInput,
        *,
        start: int = 0,
        stop: int | None = None,
    ) -> Iterator[int]:
        """Yield ids deterministically."""
        backend = backend_for(self)

        if ids is not None:
            return (int(i) for i in ids)

        with lock_for(self):
            n_items = int(backend.get_n_items())  # type: ignore[attr-defined]
        s = int(start)
        if s < 0:
            raise ValueError("start must be >= 0")
        e = n_items if stop is None else int(stop)
        if e < s:
            raise ValueError("stop must be >= start")
        e = min(e, n_items)
        return iter(range(s, e))

    def _ndarray_expected_rows(
        self,
        ids: IdsInput,
        *,
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
    ) -> int:
        """Return expected number of rows deterministically."""
        backend = backend_for(self)
        if ids is None:
            with lock_for(self):
                n_items = int(backend.get_n_items())  # type: ignore[attr-defined]
            s = int(start)
            e = n_items if stop is None else int(stop)
            e = min(e, n_items)
            if e < s:
                raise ValueError("stop must be >= start")
            return max(0, e - s)

        if isinstance(ids, Sequence):
            return len(ids)

        if n_rows is None:
            raise TypeError("For non-sized iterables, pass n_rows explicitly.")
        n = int(n_rows)
        if n < 0:
            raise ValueError("n_rows must be >= 0")
        return n

    def _ndarray_infer_f(self, first_vec: Sequence[float] | None) -> int:
        """Infer index dimension ``f`` deterministically."""
        f = int(getattr(self, "f", 0) or 0)
        if f > 0:
            return f
        return 0 if first_vec is None else len(first_vec)

    def _ndarray_materialize_dense(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        dtype: Any = np.float32,
        n_rows: int | None = None,
        return_ids: bool = False,
        validate_vector_len: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Materialize selected vectors into a dense NumPy array."""
        backend = backend_for(self)

        n = self._ndarray_expected_rows(ids, start=start, stop=stop, n_rows=n_rows)
        it = self._ndarray_iter_ids(ids, start=start, stop=stop)

        try:
            first_id = next(it)
        except StopIteration as e:
            if n != 0:
                raise ValueError("ids yielded fewer items than expected n_rows.") from e
            f = int(getattr(self, "f", 0) or 0)
            X0 = np.empty((0, int(f)), dtype=dtype)  # noqa: N806
            ids0 = np.empty((0,), dtype=np.int64) if return_ids else None
            return X0, ids0

        with lock_for(self):
            first_vec = backend.get_item_vector(int(first_id))  # type: ignore[attr-defined]
        f = self._ndarray_infer_f(first_vec)

        if n == 0:
            raise ValueError("Selection yielded at least one id but n_rows is 0.")

        X = np.empty((n, int(f)), dtype=dtype)
        ids_out = np.empty((n,), dtype=np.int64) if return_ids else None

        v0 = np.asarray(first_vec, dtype=dtype)
        if validate_vector_len and v0.shape != (int(f),):
            raise ValueError(
                f"Vector length mismatch for id {int(first_id)}: expected {int(f)}, got {tuple(v0.shape)}."
            )
        X[0] = v0
        if ids_out is not None:
            ids_out[0] = int(first_id)

        r = 1
        for item_id in it:
            if r >= n:
                raise ValueError("ids yielded more items than expected n_rows.")
            with lock_for(self):
                raw_vec = backend.get_item_vector(int(item_id))  # type: ignore[attr-defined]
            vec = np.asarray(raw_vec, dtype=dtype)
            if validate_vector_len and vec.shape != (int(f),):
                raise ValueError(
                    f"Vector length mismatch for id {int(item_id)}: expected {int(f)}, got {tuple(vec.shape)}."
                )
            X[r] = vec
            if ids_out is not None:
                ids_out[r] = int(item_id)
            r += 1

        if r != n:
            raise ValueError("ids yielded fewer items than expected n_rows.")

        return X, ids_out

    # ------------------------------------------------------------------
    # Public: batch add
    # ------------------------------------------------------------------
    def add_items(  # noqa: PLR0912
        self,
        X: Any,
        ids: Sequence[int] | Iterable[int] | None = None,
        *,
        start_id: int | None = None,
        accept_sparse: SparsePolicy = "error",
        ensure_all_finite: FinitePolicy = True,
        copy: bool = False,
        dtype: Any = np.float32,
        order: Literal["C", "F", "A", "K"] = "C",
        check_unique_ids: bool = True,
    ) -> np.ndarray:
        """
        Add many vectors to the index.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vectors to add.
        ids : array-like of shape (n_samples,), optional
            Explicit integer ids. If omitted, ids are allocated as a contiguous
            range starting at ``start_id`` (or ``get_n_items()`` at call time).
        start_id : int, optional
            Starting id used when ``ids`` is None. If None, defaults to
            ``backend.get_n_items()`` at call time.
        accept_sparse : {'error', 'toarray'}, default='error'
            Sparse input handling. ``'toarray'`` densifies SciPy sparse inputs
            explicitly. Any other sparse behavior raises.
        ensure_all_finite : bool or 'allow-nan', default=True
            Finiteness validation policy.
        copy : bool, default=False
            If True, copy the validated dense array before adding.
        dtype : numpy dtype, default=numpy.float32
            Dtype passed to the backend.
        order : {'C', 'F', 'A', 'K'}, default='C'
            Memory order used when coercing ``X``.
        check_unique_ids : bool, default=True
            If True, require ids to be unique.

        Returns
        -------
        ids_out : numpy.ndarray of shape (n_samples,)
            The ids that were added, as ``int64``.

        Raises
        ------
        RuntimeError
            If the backend indicates the index is built.
        TypeError
            If sparse input is given while ``accept_sparse='error'``.
        ValueError
            If ``X`` is not 2D, feature dimensions mismatch ``f``, ids are
            invalid, or finiteness policy is violated.

        Notes
        -----
        This method is deterministic: ids are generated predictably and vectors
        are added in row order.

        See Also
        --------
        get_item_vectors : Fetch vectors by id selection.
        to_numpy : Export vectors as a dense NumPy array.
        """
        self._ndarray_require_unbuilt()

        backend = backend_for(self)
        Xv = _as_dense_2d_float(  # noqa: N806
            X,
            dtype=dtype,
            order=order,
            copy=copy,
            accept_sparse=accept_sparse,
            ensure_all_finite=ensure_all_finite,
        )
        n_samples, n_features = map(int, Xv.shape)
        if n_samples == 0:
            return np.empty((0,), dtype=np.int64)
        if n_features <= 0:
            raise ValueError("X must have at least one feature (n_features > 0).")

        f = int(getattr(self, "f", 0) or 0)
        if f > 0 and int(n_features) != f:
            raise ValueError(
                f"X has {int(n_features)} features, but index dimension f={f}."
            )

        # sklearn-style fitted metadata (safe even without MetaMixin)
        try:
            object.__setattr__(self, "n_features_in_", int(n_features))
        except Exception:
            with contextlib.suppress(Exception):
                self.n_features_in_ = int(n_features)

        # Feature names from pandas-like inputs (explicit: only when columns exists).
        if hasattr(X, "columns"):
            try:
                cols = list(getattr(X, "columns", []))
                if len(cols) == int(n_features):
                    names = np.asarray([str(c) for c in cols], dtype=object)
                    try:
                        object.__setattr__(self, "feature_names_in_", names)
                    except Exception:
                        with contextlib.suppress(Exception):
                            self.feature_names_in_ = names
            except Exception:
                pass

        if ids is None:
            if start_id is None:
                with lock_for(self):
                    base = int(backend.get_n_items())  # type: ignore[attr-defined]
            else:
                base = int(start_id)
            if base < 0:
                raise ValueError("start_id must be >= 0")
            ids_arr = np.arange(base, base + n_samples, dtype=np.int64)
        else:
            ids_arr = _coerce_ids(ids, n_samples=n_samples)

        if check_unique_ids and int(np.unique(ids_arr).shape[0]) != int(n_samples):
            raise ValueError("ids must be unique")

        with lock_for(self):
            for item_id, vec in zip(ids_arr, Xv):
                backend.add_item(int(item_id), vec)  # type: ignore[attr-defined]

        return ids_arr

    # ------------------------------------------------------------------
    # Public: export
    # ------------------------------------------------------------------
    def get_item_vectors(
        self,
        ids: IdsInput = None,
        *,
        dtype: Any = np.float32,
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        return_ids: bool = False,
        validate_vector_len: bool = True,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Fetch many vectors as a dense NumPy array.

        Parameters
        ----------
        ids : sequence of int or iterable of int, optional
            Ids to fetch. If None, selects ``range(start, stop or n_items)``.
        dtype : numpy dtype, default=numpy.float32
            Output dtype.
        start, stop : int, optional
            Range selection used when ``ids`` is None.
        n_rows : int, optional
            Required when ``ids`` is a non-sized iterable (e.g., generator).
        return_ids : bool, default=False
            If True, also return the realized ids (int64) in row order.
        validate_vector_len : bool, default=True
            If True, verify every fetched vector has length ``f``.

        Returns
        -------
        X : numpy.ndarray of shape (n_rows, f)
            Dense matrix of vectors.
        ids_out : numpy.ndarray of shape (n_rows,), optional
            Returned when ``return_ids=True``.

        Raises
        ------
        ValueError
            If the id selection is inconsistent or vectors have unexpected length.
        TypeError
            If ``ids`` is a non-sized iterable and ``n_rows`` is not provided.

        See Also
        --------
        to_numpy : Dense NumPy export alias.
        iter_item_vectors : Streaming export without allocating a dense matrix.
        """
        X, ids_out = self._ndarray_materialize_dense(
            ids,
            start=start,
            stop=stop,
            dtype=dtype,
            n_rows=n_rows,
            return_ids=return_ids,
            validate_vector_len=validate_vector_len,
        )
        if return_ids:
            assert ids_out is not None  # noqa: S101
            return X, ids_out
        return X

    def to_numpy(
        self,
        ids: IdsInput = None,
        *,
        dtype: Any = np.float32,
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        validate_vector_len: bool = True,
    ) -> np.ndarray:
        """
        Export vectors to a dense NumPy array.

        Notes
        -----
        This is an alias of :meth:`get_item_vectors` with ``return_ids=False``.

        See Also
        --------
        get_item_vectors : Dense export with optional id output.
        iter_item_vectors : Streaming export.
        to_scipy_csr : Export as SciPy CSR.
        to_pandas : Export as pandas DataFrame.
        """
        return self.get_item_vectors(
            ids,
            dtype=dtype,
            start=start,
            stop=stop,
            n_rows=n_rows,
            return_ids=False,
            validate_vector_len=validate_vector_len,
        )

    def iter_item_vectors(
        self,
        ids: IdsInput = None,
        *,
        start: int = 0,
        stop: int | None = None,
        with_ids: bool = True,
        dtype: Any | None = None,
    ) -> Iterator[np.ndarray | tuple[int, np.ndarray]]:
        """
        Iterate vectors without allocating a dense matrix.

        Parameters
        ----------
        ids, start, stop
            Selection controls. See :meth:`get_item_vectors`.
        with_ids : bool, default=True
            If True, yield ``(id, vector)``. If False, yield vectors only.
        dtype : numpy dtype, optional
            If provided, cast output vectors to this dtype.

        Yields
        ------
        (id, vector) or vector
            Each vector is returned as a 1D NumPy array.

        See Also
        --------
        get_item_vectors : Dense export.
        """
        backend = backend_for(self)
        for item_id in self._ndarray_iter_ids(ids, start=start, stop=stop):
            with lock_for(self):
                raw_vec = backend.get_item_vector(int(item_id))  # type: ignore[attr-defined]
            vec = np.asarray(raw_vec, dtype=dtype)
            yield (int(item_id), vec) if with_ids else vec

    def to_scipy_csr(
        self,
        ids: IdsInput = None,
        *,
        dtype: Any = np.float32,
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        validate_vector_len: bool = True,
    ) -> Any:
        """
        Export vectors as a SciPy CSR matrix.

        Returns
        -------
        X : scipy.sparse.csr_matrix
            CSR matrix with shape ``(n_rows, f)``.

        Raises
        ------
        ImportError
            If SciPy is not installed.

        See Also
        --------
        to_numpy : Dense NumPy export.
        to_pandas : Export as pandas DataFrame.
        """
        sp = _lazy_import_scipy_sparse()
        X = self.to_numpy(
            ids,
            dtype=dtype,
            start=start,
            stop=stop,
            n_rows=n_rows,
            validate_vector_len=validate_vector_len,
        )
        return sp.csr_matrix(X)

    def to_pandas(
        self,
        ids: IdsInput = None,
        *,
        dtype: Any = np.float32,
        start: int = 0,
        stop: int | None = None,
        n_rows: int | None = None,
        id_location: Literal["index", "column", "both", "none"] = "index",
        id_name: str = "id",
        columns: Sequence[str] | None = None,
        validate_vector_len: bool = True,
    ) -> Any:
        """
        Export vectors to a pandas ``DataFrame``.

        Parameters
        ----------
        ids, start, stop, n_rows
            Selection controls. See :meth:`get_item_vectors`.
        dtype : numpy dtype, default=numpy.float32
            Output dtype.
        id_location : {'index', 'column', 'both', 'none'}, default='index'
            Where to place ids in the output.
        id_name : str, default='id'
            Name used for the id column / index.
        columns : sequence of str, optional
            Column names for vector dimensions. If None, uses ``feature_names_in_``
            when present and length matches ``f``; otherwise uses
            ``feature_0..feature_{f-1}``.
        validate_vector_len : bool, default=True
            If True, verify every fetched vector has length ``f``.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with shape ``(n_rows, f)`` plus optional id metadata.

        Raises
        ------
        ImportError
            If pandas is not installed.
        ValueError
            If ``id_location`` is invalid or ``columns`` length mismatches ``f``.

        See Also
        --------
        to_numpy : Dense NumPy export.
        to_scipy_csr : Export as SciPy CSR.
        """
        if id_location not in {"index", "column", "both", "none"}:
            raise ValueError(
                "id_location must be one of {'index','column','both','none'}"
            )

        pd = _lazy_import_pandas()
        need_ids = id_location != "none"
        X, ids_out = self._ndarray_materialize_dense(
            ids,
            start=start,
            stop=stop,
            dtype=dtype,
            n_rows=n_rows,
            return_ids=need_ids,
            validate_vector_len=validate_vector_len,
        )
        f = int(X.shape[1])

        if columns is None:
            feature_names_in_ = getattr(self, "feature_names_in_", None)
            if feature_names_in_ is not None:
                try:
                    names = [str(c) for c in list(feature_names_in_)]
                except Exception:
                    names = []
                cols = names if len(names) == f else [f"feature_{j}" for j in range(f)]
            else:
                cols = [f"feature_{j}" for j in range(f)]
        else:
            cols = list(columns)
            if len(cols) != f:
                raise ValueError(f"`columns` must have length {f}, got {len(cols)}.")

        df = pd.DataFrame(X, columns=cols)

        if need_ids:
            assert ids_out is not None  # noqa: S101
            ids64 = ids_out.astype("int64", copy=False)
            if id_location in {"column", "both"}:
                df.insert(0, id_name, ids64)
            if id_location in {"index", "both"}:
                df.index = pd.Index(ids64, name=id_name)

        return df
