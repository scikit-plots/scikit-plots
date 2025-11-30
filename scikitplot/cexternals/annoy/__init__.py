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
Spotify Annoy (Approximate Nearest Neighbors Oh Yeah)
-----------------------------------------------------

A high-dimensional Approximate Nearest Neighbors (ANN) library implemented in C++
with Python bindings, optimized for memory usage, speed, and large-scale disk-based
indices that can be shared across processes via memory mapping (mmap).

Annoy is designed to find points in space that are closest to a given query point
using different distance metrics. It is particularly efficient for high-dimensional
vector data such as embeddings (word, image, or user vectors).

Key Features:

- Build trees for high-dimensional data using multiple distance metrics.
- Serialize/deserialize indexes to bytes or disk for persistence.
- Support for memory-mapped indexes for read-only sharing across processes.
- Optional multi-threaded building for faster index construction.
- Provides approximate nearest neighbors with tunable accuracy vs speed tradeoff.

Examples
--------
>>> from annoy import AnnoyIndex
>>> f = 40  # vector dimensionality
>>> t = AnnoyIndex(f, 'angular')  # Length of item vector and metric
>>> t.add_item(0, [1]*f)
>>> t.build(10)  # Build 10 trees
>>> t.get_nns_by_item(0, 1)  # Find nearest neighbor

References
----------
.. [1] `Spotify AB. (2013, Feb 20).
   "Approximate Nearest Neighbors Oh Yeah"
   Github. https://pypi.org/project/annoy
   <https://pypi.org/project/annoy>`_
"""

# This module is a dummy wrapper around the underlying C++ module.
from .annoylib import Annoy  # keep for doc
# from .annoylib import Annoy as AnnoyIndex  # alias of Annoy

from ._index import Index
from ._index import Index as AnnoyIndex  # alias of Annoy

# Define the annoy version
# https://github.com/spotify/annoy/blob/main/setup.py
__version__ = "8a7e82cb537053926b0ac6ec132b9ccc875af40c"  # Ref Branch, Tag, or Commit SHA
__author__ = "Erik Bernhardsson"
__author_email__ = "mail@erikbern.com"
__git_hash__  = "8a7e82cb537053926b0ac6ec132b9ccc875af40c"

__all__ = [
    'Annoy',
    'AnnoyIndex',
    'Index',
]
