# scikitplot/annoy/__init__.pyi
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore[]

from ..cexternals._annoy import Annoy, AnnoyIndex, annoylib  # low-level C-extension backend
from ._base import Index  # extended python-api derived annoylib.Annoy legacy c-api

from ._mixins._io import IndexIOMixin
from ._mixins._meta import MetaMixin
from ._mixins._ndarray import NDArrayMixin
from ._mixins._pickle import CompressMode, PickleMixin, PickleMode
from ._mixins._plotting import PlottingMixin
from ._mixins._vectors import VectorOpsMixin

__version__: str
__author__: str
__author_email__: str
__git_hash__: str

__all__: list[str] = [
    "Annoy",
    "AnnoyIndex",
    "CompressMode",
    "Index",
    "IndexIOMixin",
    "MetaMixin",
    "NDArrayMixin",
    "PickleMixin",
    "PickleMode",
    "PlottingMixin",
    "VectorOpsMixin",
]
