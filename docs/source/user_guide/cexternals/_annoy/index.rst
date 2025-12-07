

.. _cexternals-annoy-index:
======================================================================
ANNoy (experimental)
======================================================================

This module contains some functions related to :py:mod:`~._annoy` under
:py:mod:`~.cexternals` and extended to :py:mod:`~.annoy`.

.. seealso::
    * :ref:`annoy-index`
    * https://github.com/spotify/annoy
    * https://pypi.org/project/annoy


.. seealso::
    * :py:obj:`~scikitplot.annoy.Index.from_low_level`
    * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled

TL;DR
------------
- What it is: C++ library with Python bindings, “Approximate Nearest Neighbors Oh Yeah” [1]_, [2]_
- Origin: Developed at Spotify
- Purpose: Search for points in space close to a query point
- Index type: Forest of random projection trees
- Memory: Supports large read-only file-based indexes, memory-mapped to allow sharing across processes
- Installation: pip install --user annoy or clone C++ repo
- Use: Fast approximate nearest neighbor search, especially for high-dimensional vector spaces
- Example: Python snippet using AnnoyIndex from your provided info
- Distance metrics: Euclidean, Manhattan, cosine, Hamming, dot (inner product)
- Trade-offs / tuning: n_trees, search_k parameters for balancing speed vs. accuracy

Introduction
------------
ANNoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings
to search for points in space that are close to a given query point. It also creates
large read-only file-based data structures that are memory-mapped (mmap) into memory,
so that many processes may share the same data.

Background
----------
There are several libraries for nearest neighbor search. ANNoy is almost as fast as
the fastest libraries, but it has a distinctive feature: it can use static files as
indexes. This allows indexes to be shared across processes. Index creation is
decoupled from loading, so indexes can be passed as files and mapped into memory quickly.
ANNoy minimizes memory footprint, making indexes relatively small.

ANNoy was developed at Spotify for music recommendations. After matrix factorization,
every user/item can be represented as a vector in f-dimensional space. ANNoy helps
search for similar users/items efficiently, even with millions of vectors in
high-dimensional spaces. It was originally built by Erik Bernhardsson during Hack Week.

Installation
------------
Python version:

    pip install --user annoy

C++ version:

    Clone the repo and include:

        #include "annoylib.h"

Features
--------
- Distance metrics: Euclidean, Manhattan, cosine, Hamming, Dot (Inner) Product
- Cosine distance = Euclidean distance of normalized vectors: sqrt(2 - 2 * cos(u, v))
- Works best with <100 dimensions but performs well up to ~1,000 dimensions
- Small memory usage
- Supports memory sharing across processes
- Index creation is separate from lookup; no additional items can be added after building
- Native Python support (tested with 2.7, 3.6, 3.7)
- On-disk index building for datasets too large to fit in RAM (contributed by Rene Hollander)

Python Example
--------------
.. code-block:: python

    # from annoy import AnnoyIndex
    from scikitplot.annoy import Annoy, AnnoyBase, AnnoyIndex, Index
    import random

    f = 40  # Length of item vector that will be indexed
    t = AnnoyIndex(f, 'angular')
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
        t.add_item(i, v)

    t.build(10)  # 10 trees
    t.save('test.ann')

    u = AnnoyIndex(f, 'angular')
    u.load('test.ann')  # memory-mapped
    print(u.get_nns_by_item(0, 1000))

Notes
-----
- Accepts integer identifiers; memory allocated up to max(id)+1
- C++ API is similar: include "annoylib.h"
- No bounds checking is performed
- Angular distance = Euclidean distance of normalized vectors
- Hamming and Dot Product distances optimized with low-level implementations

Trade-offs
----------
- `n_trees` (build-time) affects index size and precision
- `search_k` (query-time) affects search accuracy vs speed
- Prefaulting pages affects load time and early query performance

Supported Platforms and Bindings
--------------------------------
- Python, C++, R, Java (cosine only), Scala, Ruby, Go (experimental), Lua, Rust, .NET, Node
- Available via conda for Linux, OS X, and Windows

Testing
-------
Run tests using:

    python setup.py nosetests

References
----------
.. [1] http://en.wikipedia.org/wiki/Nearest_neighbor_search#Approximate_nearest_neighbor
