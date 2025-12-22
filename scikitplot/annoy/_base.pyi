# scikitplot/annoy/_base.pyi

from typing import Any, ClassVar  # noqa: F401

from typing_extensions import Literal, Self, TypeAlias

from ..cexternals._annoy import Annoy, AnnoyIndex  # noqa: F401
from ._mixins._io import IndexIOMixin, PickleIOMixin
from ._mixins._manifest import ManifestMixin
from ._mixins._ndarray import NDArrayExportMixin
from ._mixins._pickle import CompressMode, PickleMixin, PickleMode  # noqa: F401
from ._mixins._plotting import PlottingMixin
from ._mixins._vectors import VectorOpsMixin

# --- Allowed metric literals (simple type hints) ---
# AnnoyMetric: TypeAlias = Literal["angular", "euclidean", "manhattan", "dot", "hamming"]
AnnoyMetric: TypeAlias = Literal[
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

__all__: list[str, ...] = ["Index"]

class Index(
    Annoy,
    ManifestMixin,
    IndexIOMixin,
    PickleIOMixin,
    PickleMixin,
    VectorOpsMixin,
    NDArrayExportMixin,
    PlottingMixin,
):
    # __slots__: ClassVar[tuple[str, ...]] = ()

    def _low_level(self) -> Any: ...
    @property
    def annoy(self) -> Annoy: ...
    @classmethod
    def from_low_level(cls, obj: Annoy, *, prefault: bool | None = None) -> Self: ...
