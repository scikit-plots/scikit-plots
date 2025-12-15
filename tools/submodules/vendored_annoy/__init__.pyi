# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the annoy project.
# https://github.com/spotify/annoy/blob/main/annoy/__init__.pyi

from . import annoylib
from .annoylib import Annoy  # low-level C-extension type, simple legacy c-api

AnnoyIndex = Annoy  # alias of Annoy Index c-api

__version__: str
__author__: str
__author_email__: str
__git_hash__: str

__all__: list[str]
