# scikitplot/annoy/__init__.py
# Copyright (c) 2013 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# Authors: Spotify AB
# SPDX-License-Identifier: Apache-2.0

"""
Public Annoy Python API for scikitplot.

Spotify ANNoy [0]_ (Approximate Nearest Neighbors Oh Yeah).

This package exposes **two layers**:

Exports:

1. Low-level C-extension types copied from Spotify's *annoy* project:
   :class:`~scikitplot.cexternals._annoy.Annoy` and :class:`~scikitplot.cexternals._annoy.AnnoyIndex`.

2. A high-level, mixin-composed wrapper :class:`~scikitplot.annoy.Index` that:
   - forwards the complete low-level API deterministically,
   - adds versioned manifest import/export,
   - provides explicit index I/O names (``save_index`` / ``load_index``),
   - provides safe Python-object persistence helpers (pickling),
   - adds optional NumPy export and plotting utilities.

Notes
-----
This module intentionally avoids side effects at import time (no implicit NumPy
or matplotlib imports).

.. seealso::
    * :ref:`ANNoy <annoy-index>`
    * :ref:`cexternals/ANNoy (experimental) <cexternals-annoy-index>`
    * https://github.com/spotify/annoy
    * https://pypi.org/project/annoy

See Also
--------
scikitplot.cexternals._annoy
    Low-level C-extension backend.
scikitplot.annoy.Index
    High-level wrapper composed from mixins.

References
----------
.. [0] `Spotify AB. (2013, Feb 20). "Approximate Nearest Neighbors Oh Yeah"
   Github. https://pypi.org/project/annoy <https://pypi.org/project/annoy>`_

Examples
--------
>>> import random
>>> random.seed(0)

>>> # from annoy import AnnoyIndex
>>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
>>> from scikitplot.annoy import Annoy, AnnoyIndex, Index

>>> f = 40  # vector dimensionality
>>> t = Index(f, "angular")  # same constructor as the low-level backend
>>> t.add_item(0, [1] * f)
>>> t.build(10)  # Build 10 trees
>>> t.get_nns_by_item(0, 1)  # Find nearest neighbor
"""

# https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
# https://www.devasking.com/issue/passing-arguments-to-tpnew-and-tpinit-from-subtypes-in-python-c-api

from __future__ import annotations

from importlib import metadata as _metadata  # noqa: F401

# --- Low-level backend (C++ Annoy) -----------------------------
from ..cexternals._annoy import (  # noqa: F401
    Annoy,
    AnnoyIndex,
    annoylib,  # type: ignore[]
)

# --- High-level Python API ------------------------------------
from ._base import Index  # extended python-api derived annoylib.Annoy legacy c-api

# Mixins are intentionally exported for advanced users who want to build their
# own wrapper types around the same C-extension backend.
from ._mixins._io import IndexIOMixin
from ._mixins._meta import MetaMixin
from ._mixins._ndarray import NDArrayMixin
from ._mixins._pickle import CompressMode, PickleMixin, PickleMode
from ._mixins._plotting import PlottingMixin
from ._mixins._vectors import VectorOpsMixin

# Define the annoy version
# https://github.com/spotify/annoy/blob/main/setup.py
__version__ = "2.0.0+git.20251130.8a7e82cb537053926b0ac6ec132b9ccc875af40c"  # Ref Branch, Tag, or Commit SHA
__author__ = "Erik Bernhardsson"
__author_email__ = "mail@erikbern.com"
__git_hash__ = "8a7e82cb537053926b0ac6ec132b9ccc875af40c"

# try:
#     __version__ = _metadata.version("scikitplot")
# except _metadata.PackageNotFoundError:  # pragma: no cover
#     __version__ = "0+unknown"

__all__ = [
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
