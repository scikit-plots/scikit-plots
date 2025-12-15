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

.. _annoy-index:
ANNoy
======================================================================

Examples relevant to the :py:mod:`~.annoy` module with  :py:mod:`~._annoy`.

:py:mod:`~.annoy` is a Python module that provides high-performance approximate nearest neighbor search in Python.

.. seealso::
   * `github: ANNoy based on random projection (hyperplane) trees <https://github.com/spotify/annoy>`__
   * `pypi: ANNoy based on random projection (hyperplane) method <https://pypi.org/project/annoy>`__
   * `github: Voyager based on HNSW algorithm (hnswlib) <https://github.com/spotify/voyager>`__
   * `pypi: Voyager based on HNSW algorithm (hnswlib) <https://pypi.org/project/voyager>`__
   * `github: HNSW implementation Header-only C++/python <https://github.com/nmslib/hnswlib>`__
   * `pypi: HNSW implementation Header-only C++/python <https://pypi.org/project/hnswlib>`__

.. seealso::
   * :ref:`cexternals-annoy-index`
   * :py:obj:`~scikitplot.annoy.Index.from_low_level`
   * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled

Python Example
--------------
.. code-block:: python

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
