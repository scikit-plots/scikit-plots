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

.. seealso::

   * :ref:`cex-annoy-index`
   * https://pypi.org/project/annoy
