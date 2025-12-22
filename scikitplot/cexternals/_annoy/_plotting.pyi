# scikitplot/cexternals/_annoy/_plotting.pyi
# fmt: off
# ruff: noqa
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

"""
Typing stubs for :mod:`scikitplot.cexternals._annoy._plotting`.

This stub mirrors the public API of the corresponding ``_plotting.py`` module.
Docstrings and examples live in the implementation module.

Notes
-----
- Matplotlib types are referenced only under ``TYPE_CHECKING`` to avoid
  importing plotting backends at import time.
"""

from __future__ import annotations

from os.path import abspath, dirname, join
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, TypeAlias, Union, overload, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any  # type: ignore[misc,assignment]

# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------
FILE_DIR: str
DATA_DIR: str

MACOSKO_COLORS: Mapping[str, str]
ZEISEL_COLORS: Mapping[str, str]
MOUSE_10X_COLORS: Mapping[int, str]

__all__: list[str]

# -----------------------------------------------------------------------------
# Shared typing helpers
# -----------------------------------------------------------------------------
NDArray: TypeAlias = npt.NDArray[Any]
SparseMatrix: TypeAlias = sp.spmatrix
DenseOrSparse: TypeAlias = Union[NDArray, SparseMatrix]

class AnnoyVectorsLike(Protocol):
    """Minimum interface required to materialize vectors from an Annoy-style index."""
    def get_n_items(self) -> int: ...
    def get_item_vector(self, i: int) -> Sequence[float]: ...

class AnnoyKNNLike(AnnoyVectorsLike, Protocol):
    """Minimum interface required to query neighbors from an Annoy-style index."""
    def get_nns_by_item(
        self,
        i: int,
        n: int,
        search_k: int = -1,
        include_distances: bool = False,
    ) -> Sequence[int] | Tuple[Sequence[int], Sequence[float]]: ...

# -----------------------------------------------------------------------------
# Numeric utilities
# -----------------------------------------------------------------------------
@overload
def calculate_cpm(x: NDArray, axis: int = 1) -> NDArray: ...
@overload
def calculate_cpm(x: SparseMatrix, axis: int = 1) -> SparseMatrix: ...
def calculate_cpm(x: DenseOrSparse, axis: int = 1) -> DenseOrSparse: ...
@overload
def log_normalize(data: NDArray) -> NDArray: ...
@overload
def log_normalize(data: SparseMatrix) -> SparseMatrix: ...
def log_normalize(data: DenseOrSparse) -> DenseOrSparse: ...

def pca(x: DenseOrSparse, n_components: int = 50) -> NDArray: ...

def select_genes(
    data: Any,
    threshold: float = 0,
    atleast: int = 10,
    yoffset: float = 0.02,
    xoffset: float = 5,
    decay: float = 1,
    n: Optional[int] = None,
    plot: bool = True,
    markers: Optional[Sequence[int]] = None,
    genes: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (6, 3.5),
    markeroffsets: Optional[Mapping[int, Tuple[float, float]]] = None,
    labelsize: int = 10,
    alpha: float = 1,
) -> npt.NDArray[np.bool_]: ...

def maxabs_scale_dense(x: NDArray, *, eps: float = 1e-12) -> NDArray: ...
def l2_normalize_rows(x: NDArray, *, eps: float = 1e-12) -> NDArray: ...

def pca_project_2d(x: NDArray, *, center: bool = True) -> NDArray: ...

def project_to_2d(
    x: NDArray,
    *,
    method: str = "pca",
    dims: Tuple[int, int] = (0, 1),
    center: bool = True,
) -> NDArray: ...

# -----------------------------------------------------------------------------
# Annoy extraction + plotting helpers
# -----------------------------------------------------------------------------
def annoy_index_to_array(
    index: AnnoyVectorsLike,
    ids: Optional[Sequence[int]] = None,
    *,
    dtype: Any = np.float32,
) -> Tuple[NDArray, npt.NDArray[np.int64]]: ...

def plot(
    x: NDArray,
    y: Sequence[Any],
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    draw_legend: bool = True,
    draw_centers: bool = False,
    draw_cluster_labels: bool = False,
    colors: Optional[Mapping[Any, Any]] = None,
    legend_kwargs: Optional[Mapping[str, Any]] = None,
    label_order: Optional[Sequence[Any]] = None,
    *,
    figsize: Tuple[float, float] = (8, 8),
    axis_off: bool = True,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    center_kwargs: Optional[Mapping[str, Any]] = None,
    text_kwargs: Optional[Mapping[str, Any]] = None,
    legend_handle_kwargs: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Axes: ...

def evaluate_embedding(
    embedding: NDArray,
    labels: NDArray,
    projection_embedding: Optional[NDArray] = None,
    projection_labels: Optional[NDArray] = None,
    sample: Optional[int] = None,
    random_state: Optional[object] = None,
) -> float: ...

def plot_annoy_index(
    index: AnnoyVectorsLike,
    labels: Optional[Sequence[Any]] = None,
    *,
    ids: Optional[Sequence[int]] = None,
    projection: str = "pca",
    dims: Tuple[int, int] = (0, 1),
    center: bool = True,
    maxabs: bool = False,
    l2_normalize: bool = False,
    dtype: Any = np.float32,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    plot_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[NDArray, npt.NDArray[np.int64], Axes]: ...

def plot_annoy_knn_edges(
    index: AnnoyKNNLike,
    y2: NDArray,
    *,
    ids: Optional[Sequence[int]] = None,
    k: int = 5,
    search_k: int = -1,
    ax: Optional[Axes] = None,
    line_kwargs: Optional[Mapping[str, Any]] = None,
    undirected: bool = True,
    axis_off=True,
) -> Axes: ...
