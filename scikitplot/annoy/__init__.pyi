# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the annoy project.
# https://github.com/spotify/annoy/blob/main/annoy/__init__.pyi

from ..cexternals._annoy import Annoy, AnnoyIndex  # low-level C-extension type, simple legacy c-api
from ._base import Index  # extended python-api derived annoylib.Annoy legacy c-api

__version__: str
__author__: str
__author_email__: str
__git_hash__: str

__all__: list[str]
