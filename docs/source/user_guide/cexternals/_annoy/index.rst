..
  https://devguide.python.org/documentation/markup/#sections
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections
  # with overline, for parts    : ######################################################################
  * with overline, for chapters : **********************************************************************
  = for sections                : ======================================================================
  - for subsections             : ----------------------------------------------------------------------
  ^ for subsubsections          : ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  " for paragraphs              : """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

.. # https://rsted.info.ucl.ac.be/
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#paragraph-level-markup
.. # https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#footnotes
.. # https://documatt.com/restructuredtext-reference/element/admonition.html
.. # attention, caution, danger, error, hint, important, note, tip, warning, admonition, seealso
.. # versionadded, versionchanged, deprecated, versionremoved, rubric, centered, hlist

.. currentmodule:: scikitplot.cexternals._annoy

.. _cexternals-annoy-index:

======================================================================
spotify/ANNoy (experimental)
======================================================================

This page documents the Annoy [0]_ integration shipped with scikit-plots.

- High-level API: :py:mod:`~scikitplot.annoy`
- Low-level bindings: :py:mod:`~scikitplot.cexternals._annoy`

Annoy (Approximate Nearest Neighbors Oh Yeah) is a C++ library with Python bindings
for approximate nearest-neighbor search in high-dimensional vector spaces. [1]_

TL;DR
-----

- **What it is:** C++ library with Python bindings for approximate nearest-neighbor (ANN) search. [1]_
- **Origin:** Developed at Spotify (Hack Week). [1]_
- **Since:** Open sourced in 2013. [3]_
- **Index type:** Forest of random projection trees. [1]_
- **Storage:** File-based indexes can be memory-mapped (mmap) and shared across processes. [1]_
- **Tuning:** Use ``n_trees`` (build) and ``search_k`` (query) to trade accuracy for speed. [1]_
- **Metrics:** Euclidean, Manhattan, cosine (angular), Hamming, dot (inner product). [1]_

Quick start
-----------

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

    import random
    random.seed(0)

    # from annoy import AnnoyIndex
    # from scikitplot.cexternals._annoy import AnnoyIndex
    from scikitplot.annoy import AnnoyIndex

    f = 40
    t = AnnoyIndex(f, "angular")

    for i in range(1000):
        v = [random.gauss(0, 1) for _ in range(f)]
        t.add_item(i, v)

    t.build(10)            # n_trees
    t.save("test.ann")

    u = AnnoyIndex(f, "angular")
    u.load("test.ann")     # memory-mapped (mmap)
    print(u.get_nns_by_item(0, 10))

Workflow
--------

1. Create an :class:`~.AnnoyIndex` with vector length ``f`` and a metric. [1]_
2. Add items with :py:obj:`~.AnnoyIndex.add_item`. [1]_
3. Build the forest with :py:obj:`~.AnnoyIndex.build`. [1]_
4. Query with :py:obj:`~.AnnoyIndex.get_nns_by_item` or :py:obj:`~.AnnoyIndex.get_nns_by_vector`. [1]_
5. Persist with :py:obj:`~.AnnoyIndex.save` and load with :py:obj:`~.AnnoyIndex.load`. [1]_

Important rules
~~~~~~~~~~~~~~~

- Every added vector must have length ``f``.
- Add items before calling :py:obj:`~.AnnoyIndex.build`. [1]_
- After :py:obj:`~.AnnoyIndex.build`, the index is used for queries. To add more items,
  discard the forest with :py:obj:`~.AnnoyIndex.unbuild`, add items, and build again.

Persistence and sharing
-----------------------

Save and load
~~~~~~~~~~~~~

- :py:obj:`~.AnnoyIndex.save` writes the index to a file.
- :py:obj:`~.AnnoyIndex.load` memory-maps (mmap) the file for fast loading and sharing
  across processes. [1]_

Prefault (optional)
~~~~~~~~~~~~~~~~~~~

Some builds expose a ``prefault`` option for :py:obj:`~.AnnoyIndex.load`. When enabled,
the loader may aggressively fault pages into memory. This is platform dependent. [1]_

On-disk build (large datasets)
------------------------------

Annoy can build an index directly into a file on disk. This is intended for datasets
that are too large to fit into memory during index construction. [1]_

Workflow
~~~~~~~~

1. Create the index.
2. Call :py:obj:`~.AnnoyIndex.on_disk_build` **before** adding any items. [1]_
3. Add items.
4. Call :py:obj:`~.AnnoyIndex.build`.
5. Query the index, or load it from other processes with :py:obj:`~.AnnoyIndex.load`. [1]_

Important rules
~~~~~~~~~~~~~~~

- Call :py:obj:`~.AnnoyIndex.on_disk_build` **before** :py:obj:`~.AnnoyIndex.add_item`. [1]_
- After building in this mode, there is no need to call :py:obj:`~.AnnoyIndex.save`
  because the file is already the backing store. [1]_

Example
~~~~~~~

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

    import random
    from scikitplot.annoy import AnnoyIndex

    random.seed(0)
    f = 40

    a = AnnoyIndex(f, "angular")
    a.on_disk_build("big.ann")

    for i in range(1000):
        v = [random.gauss(0, 1) for _ in range(f)]
        a.add_item(i, v)

    a.build(10)

    # In another process, load the same file (mmap):
    b = AnnoyIndex(f, "angular")
    b.load("big.ann")
    print(b.get_nns_by_item(0, 10))

Tuning
------

- ``n_trees`` (build time):
  Larger values usually improve recall but increase build time and memory usage. [1]_
- ``search_k`` (query time):
  Larger values usually improve recall but make queries slower. [1]_

If ``search_k`` is not provided, Annoy uses a default based on the number of trees
and the requested neighbor count. [1]_

Practical tips
--------------

Choose stable item ids
~~~~~~~~~~~~~~~~~~~~~~

Annoy uses non-negative integer item ids and allocates storage up to ``max(id) + 1``. [1]_
If your external ids are sparse or non-numeric, keep a separate mapping to compact ids.

Multi-process serving
~~~~~~~~~~~~~~~~~~~~~

A common serving pattern is:

1. Build once and write an index file.
2. In each worker process, load the same file with :py:obj:`~.AnnoyIndex.load` (mmap) and query. [1]_

Developer notes (C++)
---------------------

``AnnoyIndex`` is not copyable (it contains atomic state). In C++14, avoid copy
initialization from temporaries. Prefer direct initialization::

    using Index = Annoy::AnnoyIndex<int, double, Annoy::Angular, Annoy::Kiss32Random,
                                    Annoy::AnnoyIndexSingleThreadedBuildPolicy>;
    Index t(f);  // C++14 compatible

See also
--------

.. seealso::
    * :ref:`ANNoy <annoy-index>`
    * :ref:`cexternals/ANNoy (experimental) <cexternals-annoy-index>`
    * https://github.com/spotify/annoy
    * https://pypi.org/project/annoy

.. seealso::
   * :py:obj:`~scikitplot.annoy.Index.from_low_level`
   * :py:mod:`pickle` (Python standard library)
   * Alternative ANN libraries:
     - https://github.com/nmslib/hnswlib
     - https://github.com/spotify/voyager

References
----------
.. [0] `Spotify AB. (2013, Feb 20). "Approximate Nearest Neighbors Oh Yeah"
   Github. https://pypi.org/project/annoy <https://pypi.org/project/annoy>`_
.. [1] https://github.com/spotify/annoy (README: API, mmap, prefault, on-disk build, tuning)
.. [2] https://pypi.org/project/annoy/ (Project description)
.. [3] https://erikbern.com/2015/05/03/annoy-now-without-boost-dependencies-and-with-python-3-support.html (History note: open sourced in 2013)
