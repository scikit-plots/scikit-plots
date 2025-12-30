# scikitplot/annoy/_base.pyi

import threading

from typing_extensions import Literal, Self, TypeAlias

from ..cexternals._annoy import Annoy
from ._mixins._io import IndexIOMixin
from ._mixins._meta import MetaMixin
from ._mixins._ndarray import NDArrayMixin
from ._mixins._pickle import PickleMixin
from ._mixins._plotting import PlottingMixin
from ._mixins._vectors import VectorOpsMixin

# --- Allowed metric literals (simple type hints) ---
# AnnoyMetric: TypeAlias = Literal["angular", "euclidean", "manhattan", "dot", "hamming"]
# Sphinx autodoc understands #: for module variables (and attributes), so your description will show up.
AnnoyMetric: TypeAlias = (
    Literal[
        "angular",
        "cosine",
        "euclidean",
        "l2",
        "lstsq",
        "manhattan",
        "l1",
        "cityblock",
        "taxicab",
        "dot",
        "@",
        ".",
        "dotproduct",
        "inner",
        "innerproduct",
        "hamming",
    ]
    | None
)

__all__: list[str] = [
    "Index",
]

class Index(
    Annoy,
    MetaMixin,
    IndexIOMixin,
    PickleMixin,
    VectorOpsMixin,
    NDArrayMixin,
    PlottingMixin,
):
    _lock: threading.RLock | None

    def _get_lock(self) -> threading.RLock: ...
    def _backend(self) -> Annoy: ...
    @property
    def backend(self) -> Annoy: ...
    @classmethod
    def from_low_level(cls, obj: Annoy, *, prefault: bool | None = None) -> Self: ...
