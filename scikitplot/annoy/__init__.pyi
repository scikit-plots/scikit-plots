# scikitplot/annoy/__init__.pyi
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore[]

from ..cexternals._annoy import Annoy, AnnoyIndex  # low-level C-extension type, simple legacy c-api
from ._base import Index  # extended python-api derived annoylib.Annoy legacy c-api

from ._mixins._io import IndexIOMixin, PickleIOMixin
from ._mixins._manifest import ManifestMixin
from ._mixins._ndarray import NDArrayExportMixin
from ._mixins._pickle import CompressMode, PickleMixin, PickleMode
from ._mixins._plotting import PlottingMixin
from ._mixins._vectors import VectorOpsMixin

__version__: str
__author__: str
__author_email__: str
__git_hash__: str

__all__: list[str, ...] = [
    "Annoy",
    "AnnoyIndex",
    "CompressMode",
    "Index",
    "IndexIOMixin",
    "ManifestMixin",
    "NDArrayExportMixin",
    "PickleIOMixin",
    "PickleMixin",
    "PickleMode",
    "PlottingMixin",
    "VectorOpsMixin",
]
