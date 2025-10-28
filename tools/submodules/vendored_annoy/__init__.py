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
High-dimensional Approximate Nearest Neighbors (ANN) queries in C++/Python optimized
for memory usage and loading/saving to disk.

Python module (written in C++) for high-dimensional approximate nearest neigbor (ANN) queries.
Spotify-Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings
to search for points in space that are close to a given query point.
It also creates large read-only file-based data structures
that are mmapped into memory so that many processes may share the same data.

References
----------
.. [1] `Spotify AB. (2013, Feb 20).
   "Approximate Nearest Neighbors Oh Yeah"
   Github. https://pypi.org/project/annoy
   <https://pypi.org/project/annoy>`_
"""

# This module is a dummy wrapper around the underlying C++ module.
from .annoylib import Annoy  # keep for doc
from .annoylib import Annoy as AnnoyIndex  # alias of Annoy

# Define the annoy version
# https://github.com/spotify/annoy/blob/main/setup.py
__version__ = "8a7e82cb537053926b0ac6ec132b9ccc875af40c"  # Ref Branch, Tag, or Commit SHA
__author__ = "Erik Bernhardsson"
__author_email__ = "mail@erikbern.com"
__git_hash__  = "8a7e82cb537053926b0ac6ec132b9ccc875af40c"

__all__ = [
    'Annoy',
    'AnnoyIndex',
]
