# scikitplot/cexternals/_annoy/__init__.py
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
High-level Python interface for the C++ ANNoy backend.

Spotify ANNoy [0]_ (Approximate Nearest Neighbors Oh Yeah).

Exports:

* Annoy       → low-level C-extension type (stable) `c-api powered new features <https://scikit-plots.github.io/dev/modules/generated/scikitplot.cexternals._annoy.Annoy.html>`_
* AnnoyIndex  → alias of Annoy (legacy AnnoyIndex name)

.. seealso::
    * :ref:`ANNoy <annoy-index>`
    * :ref:`cexternals/ANNoy (experimental) <cexternals-annoy-index>`
    * https://github.com/spotify/annoy
    * https://pypi.org/project/annoy

References
----------
.. [0] `Spotify AB. (2013, Feb 20). "Approximate Nearest Neighbors Oh Yeah"
   Github. https://pypi.org/project/annoy <https://pypi.org/project/annoy>`_

Examples
--------
>>> import random; random.seed(0)
>>> # from annoy import Annoy, AnnoyIndex
>>> from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
>>> from scikitplot.annoy import Annoy, AnnoyIndex, Index

>>> f = 40  # vector dimensionality
>>> t = AnnoyIndex(f, "angular")  # Length of item vector and metric
>>> t.add_item(0, [1] * f)
>>> t.build(10)  # Build 10 trees
>>> t.get_nns_by_item(0, 1)  # Find nearest neighbor
"""

# https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
# https://www.devasking.com/issue/passing-arguments-to-tpnew-and-tpinit-from-subtypes-in-python-c-api

from __future__ import annotations

# This module is a dummy wrapper around the underlying C++ module.
# --- Low-level backend (C++ Annoy) -----------------------------
from . import annoylib
from .annoylib import *  # Annoy low-level C-extension type, simple legacy c-api

AnnoyIndex = Annoy  # alias of Annoy Index c-api

# Define the annoy version
# https://github.com/spotify/annoy/blob/main/setup.py
__version__ = "2.0.0+git.20251130.8a7e82cb537053926b0ac6ec132b9ccc875af40c"  # Ref Branch, Tag, or Commit SHA
__author__ = "Erik Bernhardsson"
__author_email__ = "mail@erikbern.com"
__git_hash__ = "8a7e82cb537053926b0ac6ec132b9ccc875af40c"

__all__ = [
    "Annoy",
    "AnnoyIndex",
    "annoylib",
    # "__version__",
    # "__author__",
    # "__author_email__",
    # "__git_hash__",
]
