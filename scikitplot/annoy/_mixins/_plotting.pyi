# scikitplot/annoy/_mixins/_plotting.pyi
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore[]

"""
Typing stubs for :mod:`~scikitplot.annoy._mixins._plotting`.

This module is intentionally a thin layer on top of
:mod:`~scikitplot.cexternals._annoy._plotting`.

Notes
-----
- Plotting backends (e.g. Matplotlib) are imported lazily at runtime.
- Return types use ``Any`` for plotting objects to avoid importing GUI libraries.
"""

# from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from matplotlib.axes import Axes

# Re-export canonical signatures (single source of truth)
from scikitplot.cexternals._annoy._plotting import (  # noqa: F401
    AnnoyKNNLike,
    AnnoyVectorsLike,
    NDArray,
    annoy_index_to_array,
    calculate_cpm,
    evaluate_embedding,
    l2_normalize_rows,
    log_normalize,
    maxabs_scale_dense,
    pca,
    pca_project_2d,
    plot,
    plot_annoy_index,
    plot_annoy_knn_edges,
    project_to_2d,
    select_genes,
)

__all__: list[str] = [
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
    "PlottingMixin",
]


def plot(*args: Any, **kwargs: Any) -> Any: ...
def evaluate_embedding(*args: Any, **kwargs: Any) -> float: ...
def annoy_index_to_array(*args: Any, **kwargs: Any) -> tuple[NDArray[Any], NDArray[Any]]: ...
def project_to_2d(*args: Any, **kwargs: Any) -> NDArray[Any]: ...
def plot_annoy_index(*args: Any, **kwargs: Any) -> tuple[NDArray[Any], NDArray[Any], Any]: ...
def plot_annoy_knn_edges(*args: Any, **kwargs: Any) -> Any: ...
def calculate_cpm(*args: Any, **kwargs: Any) -> Any: ...
def log_normalize(*args: Any, **kwargs: Any) -> Any: ...
def pca(*args: Any, **kwargs: Any) -> NDArray[Any]: ...
def select_genes(*args: Any, **kwargs: Any) -> NDArray[Any]: ...
def maxabs_scale_dense(*args: Any, **kwargs: Any) -> NDArray[Any]: ...
def l2_normalize_rows(*args: Any, **kwargs: Any) -> NDArray[Any]: ...
def pca_project_2d(*args: Any, **kwargs: Any) -> NDArray[Any]: ...


class PlottingMixin:
    """
    Mixin that adds convenient plotting methods to high-level Annoy wrappers.

    The host class is expected to be Annoy-like (duck-typed). If the host wraps
    an internal Annoy instance, override :meth:`_plotting_backend` to return that
    backend.
    """

    @staticmethod
    def _as_2d_coords(y2: Any) -> npt.NDArray[Any]: ...

    def plot_index(
        self,
        labels: Sequence[Any] | None = ...,
        *,
        ids: Sequence[int] | None = ...,
        projection: str = ...,
        dims: tuple[int, int] = ...,
        center: bool = ...,
        maxabs: bool = ...,
        l2_normalize: bool = ...,
        dtype: Any = ...,
        ax: Any = ...,
        title: str | None = ...,
        plot_kwargs: Mapping[str, Any] | None = ...,
    ) -> tuple[NDArray[Any], NDArray[Any], Any]: ...

    def plot_knn_edges(
        self,
        y2: NDArray[Any],
        *,
        ids: Sequence[int] | None = ...,
        k: int = ...,
        search_k: int = ...,
        ax: Any = ...,
        line_kwargs: Mapping[str, Any] | None = ...,
        undirected: bool = ...,
    ) -> Any: ...
