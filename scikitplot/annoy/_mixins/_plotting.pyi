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

This module intentionally remains a thin, stable layer on top of
:mod:`~scikitplot.cexternals._annoy._plotting`.

Notes
-----
- The runtime implementation performs lazy imports to avoid requiring a plotting
  backend at import time.
- For typing, we re-export the canonical function signatures from the low-level
  module.
"""

# from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any  # type: ignore[misc,assignment]

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

__all__: list[str]


class PlottingMixin:
    """
    Mixin that adds convenient plotting methods to high-level Annoy wrappers.

    The host class is expected to be Annoy-like (duck-typed). If the host wraps
    an internal Annoy instance, override :meth:`_plotting_backend` to return that
    backend.
    """

    def _low_level(self) -> Any: ...

    def _plotting_backend(self) -> AnnoyKNNLike: ...

    @staticmethod
    def _as_2d_coords(y2: Any) -> npt.NDArray[Any]: ...

    def plot_index(
        self,
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

    def plot_knn_edges(
        self,
        y2: NDArray,
        *,
        ids: Optional[Sequence[int]] = None,
        k: int = 10,
        search_k: int = -1,
        ax: Optional[Axes] = None,
        line_kwargs: Optional[Mapping[str, Any]] = None,
        undirected: bool = True,
    ) -> Axes: ...
