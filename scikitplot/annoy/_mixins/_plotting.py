# scikitplot/annoy/_mixins/_plotting.py
"""
High-level plotting helpers for :mod:`scikitplot.annoy`.

This module is the **high-level** counterpart of
:mod:`scikitplot.cexternals._annoy._plotting`.

The project has two layers:

1. **Low-level** (C-extension + thin Python helpers)
   - :mod:`scikitplot.cexternals._annoy` and
     :mod:`scikitplot.cexternals._annoy._plotting`
2. **High-level** user API
   - :mod:`scikitplot.annoy` and composable mixins in
     :mod:`scikitplot.annoy._mixins`

To avoid duplicated implementations drifting over time, this module **delegates**
all plotting/projection helpers to the low-level implementation and only adds a
small :class:`~PlottingMixin` that high-level classes (e.g.,
``scikitplot.annoy.Index``) can inherit.

See Also
--------
scikitplot.cexternals._annoy._plotting
    Canonical implementation of plotting + deterministic projections.
scikitplot.annoy._mixins
    Mixins composed by high-level Annoy wrappers.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .._utils import backend_for, lock_for

__all__ = [  # noqa: RUF022
    # Re-exported low-level helpers (thin wrappers)
    "annoy_index_to_array",
    "calculate_cpm",
    "evaluate_embedding",
    "l2_normalize_rows",
    "log_normalize",
    "maxabs_scale_dense",
    "pca",
    "pca_project_2d",
    "plot",
    "plot_annoy_index",
    "plot_annoy_knn_edges",
    "project_to_2d",
    "select_genes",
    # High-level API
    "PlottingMixin",
]


def _cexternals_plotting():
    """
    Import :mod:`scikitplot.cexternals._annoy._plotting` lazily.

    Keeping this import inside a function ensures importing
    :mod:`scikitplot.annoy` does not immediately require a plotting backend.
    """

    from scikitplot.cexternals._annoy import _plotting as _cplot  # noqa: PLC0415

    return _cplot


# ---------------------------------------------------------------------------
# Public function shims (single source of truth lives in cexternals)
# ---------------------------------------------------------------------------


def plot(*args: Any, **kwargs: Any) -> Any:
    """See :func:`scikitplot.cexternals._annoy._plotting.plot`."""

    return _cexternals_plotting().plot(*args, **kwargs)


def evaluate_embedding(*args: Any, **kwargs: Any) -> float:
    """See :func:`scikitplot.cexternals._annoy._plotting.evaluate_embedding`."""

    return _cexternals_plotting().evaluate_embedding(*args, **kwargs)


def annoy_index_to_array(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    """See :func:`scikitplot.cexternals._annoy._plotting.annoy_index_to_array`."""

    return _cexternals_plotting().annoy_index_to_array(*args, **kwargs)


def project_to_2d(*args: Any, **kwargs: Any) -> np.ndarray:
    """See :func:`scikitplot.cexternals._annoy._plotting.project_to_2d`."""

    return _cexternals_plotting().project_to_2d(*args, **kwargs)


def plot_annoy_index(*args: Any, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    """See :func:`scikitplot.cexternals._annoy._plotting.plot_annoy_index`."""

    return _cexternals_plotting().plot_annoy_index(*args, **kwargs)


def plot_annoy_knn_edges(*args: Any, **kwargs: Any) -> Any:
    """See :func:`scikitplot.cexternals._annoy._plotting.plot_annoy_knn_edges`."""

    return _cexternals_plotting().plot_annoy_knn_edges(*args, **kwargs)


def calculate_cpm(*args: Any, **kwargs: Any) -> Any:
    """See :func:`scikitplot.cexternals._annoy._plotting.calculate_cpm`."""

    return _cexternals_plotting().calculate_cpm(*args, **kwargs)


def log_normalize(*args: Any, **kwargs: Any) -> Any:
    """See :func:`scikitplot.cexternals._annoy._plotting.log_normalize`."""

    return _cexternals_plotting().log_normalize(*args, **kwargs)


def pca(*args: Any, **kwargs: Any) -> np.ndarray:
    """See :func:`scikitplot.cexternals._annoy._plotting.pca`."""

    return _cexternals_plotting().pca(*args, **kwargs)


def select_genes(*args: Any, **kwargs: Any) -> np.ndarray:
    """See :func:`scikitplot.cexternals._annoy._plotting.select_genes`."""

    return _cexternals_plotting().select_genes(*args, **kwargs)


def maxabs_scale_dense(*args: Any, **kwargs: Any) -> np.ndarray:
    """See :func:`scikitplot.cexternals._annoy._plotting.maxabs_scale_dense`."""

    return _cexternals_plotting().maxabs_scale_dense(*args, **kwargs)


def l2_normalize_rows(*args: Any, **kwargs: Any) -> np.ndarray:
    """See :func:`scikitplot.cexternals._annoy._plotting.l2_normalize_rows`."""

    return _cexternals_plotting().l2_normalize_rows(*args, **kwargs)


def pca_project_2d(*args: Any, **kwargs: Any) -> np.ndarray:
    """See :func:`scikitplot.cexternals._annoy._plotting.pca_project_2d`."""

    return _cexternals_plotting().pca_project_2d(*args, **kwargs)


# ---------------------------------------------------------------------------
# High-level convenience mixin
# ---------------------------------------------------------------------------


class PlottingMixin:
    """
    Mixin that adds convenient plotting methods to high-level Annoy wrappers.

    The mixin assumes the *host class* is Annoy-like (implements
    ``get_n_items()``, ``get_item_vector(i)``, and ``get_nns_by_item(...)``).
    If your wrapper delegates to an internal Annoy instance, override
    :meth:`_plotting_backend` to return that backend.

    Notes
    -----
    - The underlying plotting logic is implemented in
      :mod:`scikitplot.cexternals._annoy._plotting`.
    - All methods are deterministic given index contents and parameters.

    See Also
    --------
    plot_annoy_index
        Function-level plotting helper.
    plot_annoy_knn_edges
        Function-level kNN edge overlay helper.
    """

    # NOTE: This method is intentionally tiny and explicit.
    # Subclasses that wrap/compose Annoy should override it.
    def _plotting_backend(self) -> Any:
        """
        Return the Annoy-like backend used for plotting.

        The default implementation is composition-friendly and deterministic:
        it returns :func:`~scikitplot.annoy._utils.backend_for`.

        Override this method if your wrapper uses a different delegation
        mechanism.
        """

        return backend_for(self)

    @staticmethod
    def _as_2d_coords(y2: Any) -> np.ndarray:
        """
        Validate and coerce 2D coordinates.

        Parameters
        ----------
        y2
            Array-like of shape ``(n_samples, 2)``.

        Returns
        -------
        y2_arr
            NumPy array view/copy of shape ``(n_samples, 2)``.

        Notes
        -----
        This helper exists to keep :meth:`plot_knn_edges` robust and deterministic
        for both NumPy arrays and array-like inputs.

        See Also
        --------
        plot_knn_edges
            Uses this helper to validate coordinates.
        """

        y2_arr = np.asarray(y2)
        if y2_arr.ndim != 2 or y2_arr.shape[1] != 2:  # noqa: PLR2004
            raise ValueError(
                "y2 must be a 2D array of shape (n_samples, 2); "
                f"got shape={getattr(y2_arr, 'shape', None)!r}"
            )
        return y2_arr

    def plot_index(
        self,
        labels: Sequence[Any] | None = None,
        *,
        ids: Sequence[int] | None = None,
        projection: str = "pca",
        dims: tuple[int, int] = (0, 1),
        center: bool = True,
        maxabs: bool = False,
        l2_normalize: bool = False,
        dtype: Any = np.float32,
        ax: Any = None,
        title: str | None = None,
        plot_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, Any]:
        """
        Plot this index as a 2D scatter plot.

        This is a thin wrapper around :func:`plot_annoy_index` that uses
        :meth:`_plotting_backend`.

        Parameters
        ----------
        labels, ids, projection, dims, center, maxabs, l2_normalize, dtype, ax, title, plot_kwargs
            See :func:`plot_annoy_index`.

        Returns
        -------
        y2, ids_out, ax
            See :func:`plot_annoy_index`.

        Notes
        -----
        - This method does not mutate the index.
        - Plotting backends (e.g. Matplotlib) are imported lazily and are only
          required when this method is called.
        - The returned ``ids_out`` corresponds to the item id for each row in
          ``y2``.

        See Also
        --------
        plot_annoy_index
            Low-level plotting helper this method delegates to.
        plot_knn_edges
            Overlay kNN edges on the returned 2D coordinates.

        Examples
        --------
        >>> import numpy as np
        >>> import scikitplot.annoy as skann
        >>> idx = skann.Index(f=10, metric="angular")
        >>> # ... add items & build ...
        >>> labels = np.zeros(idx.get_n_items(), dtype=int)
        >>> y2, ids, ax = idx.plot_index(labels=labels, projection="pca")
        """

        backend = self._plotting_backend()
        with lock_for(self):
            return plot_annoy_index(
                backend,
                labels=labels,
                ids=ids,
                projection=projection,
                dims=dims,
                center=center,
                maxabs=maxabs,
                l2_normalize=l2_normalize,
                dtype=dtype,
                ax=ax,
                title=title,
                plot_kwargs=plot_kwargs,
            )

    def plot_knn_edges(
        self,
        y2: np.ndarray,
        *,
        ids: Sequence[int] | None = None,
        k: int = 10,
        search_k: int = -1,
        ax: Any = None,
        line_kwargs: Mapping[str, Any] | None = None,
        undirected: bool = True,
    ) -> Any:
        """
        Overlay kNN edges onto an existing 2D index plot.

        This is a thin wrapper around :func:`plot_annoy_knn_edges` that uses
        :meth:`_plotting_backend`.

        Parameters
        ----------
        y2, ids, k, search_k, ax, line_kwargs, undirected
            See :func:`plot_annoy_knn_edges`.

        Returns
        -------
        ax
            The axes that were drawn on.

        Notes
        -----
        - ``y2`` must represent 2D coordinates with shape ``(n_samples, 2)``.
        - If ``ids`` is provided, it must have length ``n_samples``.
        - This method does not mutate the index; it only performs neighbor
          queries to draw edges.

        See Also
        --------
        plot_annoy_knn_edges
            Low-level edge overlay helper this method delegates to.
        plot_index
            Computes the 2D coordinates used as input to this method.

        Examples
        --------
        >>> y2, ids, ax = idx.plot_index(labels=np.zeros(idx.get_n_items(), dtype=int))
        >>> idx.plot_knn_edges(y2, ids=ids, k=5, line_kwargs={"alpha": 0.15})
        """

        y2_arr = self._as_2d_coords(y2)
        if ids is not None and len(ids) != y2_arr.shape[0]:
            raise ValueError(
                "ids must have the same length as the number of rows in y2; "
                f"len(ids)={len(ids)!r}, y2.shape={y2_arr.shape!r}"
            )

        backend = self._plotting_backend()
        with lock_for(self):
            return plot_annoy_knn_edges(
                backend,
                y2_arr,
                ids=ids,
                k=k,
                search_k=search_k,
                ax=ax,
                line_kwargs=line_kwargs,
                undirected=undirected,
            )
