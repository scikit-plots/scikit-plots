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

.. currentmodule:: scikitplot.annoy

.. _annoy-index:

======================================================================
ANNoy
======================================================================

This page documents the Annoy [0]_ integration shipped with scikit-plots.

- High-level API: :py:mod:`~scikitplot.annoy`
- Low-level bindings: :py:mod:`~scikitplot.cexternals._annoy`

High-level Python interface for the C++ Annoy backend.

This page documents :py:mod:`~scikitplot.annoy`. It provides a stable import path
and a small, user-facing API built on the low-level bindings in
:py:mod:`~scikitplot.cexternals._annoy`.

.. note::
   For backend and C-extension details, see :ref:`cexternals-annoy-index`.

Exports
--------

This module exports:

- :class:`~Annoy`:
  Low-level C-extension type (stable).
- :class:`~AnnoyIndex`:
  Public alias of the Annoy index.
- :class:`~Index`:
  High-level Python wrapper subclass (picklable).


.. grid:: 1 1 2 2

    .. grid-item-card::
        :padding: 2

        **architecture**
        ^^^
        .. toctree::
            :maxdepth: 2

            Index Architecture <annoy_index_inheritance_diagrams.rst>


Workflow
--------

1. Create an :class:`~AnnoyIndex` with a fixed vector length ``f`` and a metric.
2. Add items with :py:obj:`~AnnoyIndex.add_item`.
3. Build the forest with :py:obj:`~AnnoyIndex.build`.
4. Save and load with :py:obj:`~AnnoyIndex.save` and :py:obj:`~AnnoyIndex.load`.

Quick start
-----------

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

    import random; random.seed(0)
    # from annoy import Annoy, AnnoyIndex
    # from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
    from scikitplot.annoy import Annoy, AnnoyIndex, Index

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

* Every added vector must have length ``f``.
* Add items before calling :py:obj:`~AnnoyIndex.build`.
* Item ids are integers. Storage is allocated up to ``max(id) + 1``.

High-level wrapper: :class:`~Index`
-----------------------------------

:class:`~Index` is a Pythonic wrapper for Annoy-like objects.

It is designed for higher-level workflows where you want a Python object that is
safe to serialize and move between processes.

Mixins used by the high-level wrapper
-------------------------------------

The wrapper uses mixins :py:mod:`~scikitplot.annoy._mixins`
to keep features separate and explicit.

Further reading
---------------

.. seealso::
    * :ref:`ANNoy <annoy-index>`
    * :ref:`cexternals/ANNoy (experimental) <cexternals-annoy-index>`
    * https://github.com/spotify/annoy
    * https://pypi.org/project/annoy

.. seealso::
   * `github: ANNoy based on random projection (hyperplane) trees <https://github.com/spotify/annoy>`__
   * `pypi: ANNoy based on random projection (hyperplane) method <https://pypi.org/project/annoy>`__
   * `github: Voyager based on HNSW algorithm (hnswlib) <https://github.com/spotify/voyager>`__
   * `pypi: Voyager based on HNSW algorithm (hnswlib) <https://pypi.org/project/voyager>`__
   * `github: HNSW implementation Header-only C++/python <https://github.com/nmslib/hnswlib>`__
   * `pypi: HNSW implementation Header-only C++/python <https://pypi.org/project/hnswlib>`__

   * Python pickling: https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
   * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled

.. * Annoy overview: https://www.sandgarden.com/learn/annoy-approximate-nearest-neighbors-oh-yeah
.. * FAISS overview: https://www.sandgarden.com/learn/faiss
.. seealso::
   * Nearest neighbor search (background): https://en.wikipedia.org/wiki/Nearest_neighbor_search
   * https://www.researchgate.net/publication/386374637_Optimizing_Domain-Specific_Image_Retrieval_A_Benchmark_of_FAISS_and_Annoy_with_Fine-Tuned_Features
   * https://www.researchgate.net/publication/363234433_Analysis_of_Image_Similarity_Using_CNN_and_ANNOY
   * https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf
   * https://link.springer.com/chapter/10.1007/978-981-97-7831-7_2

References
----------
.. [0] `Spotify AB. (2013, Feb 20). "Approximate Nearest Neighbors Oh Yeah"
   Github. https://pypi.org/project/annoy <https://pypi.org/project/annoy>`_
