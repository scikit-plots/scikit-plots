.. docs/source/user_guide/annoy/annoy_index_inheritance_diagrams.rst
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

.. _annoy_index_inheritance_diagrams:

======================================================================
Annoy :class:`~Index` class Inheritance Diagrams (MRO)
======================================================================

This page shows the **class inheritance structure** for the Annoy backend and the
high-level :class:`~scikitplot.annoy.Index` facade class, using Sphinx’s built-in
inheritance diagram support (the same approach used in Matplotlib’s docs).

----------------------------------------------------------------------
Index + mixins
----------------------------------------------------------------------

.. inheritance-diagram::
   scikitplot.cexternals._annoy.Annoy
   scikitplot.annoy._mixins._meta.MetaMixin
   scikitplot.annoy._mixins._io.IndexIOMixin
   scikitplot.annoy._mixins._pickle.PickleMixin
   scikitplot.annoy._mixins._vectors.VectorOpsMixin
   scikitplot.annoy._mixins._ndarray.NDArrayMixin
   scikitplot.annoy._mixins._plotting.PlottingMixin
   scikitplot.annoy.Index
   :top-classes: scikitplot.cexternals._annoy.Annoy
   :parts: 1
   :private-bases:

.. rubric:: Reading this diagram

- :class:`~scikitplot.cexternals._annoy.Annoy` is the **low-level C-extension backend**
  and owns the actual index state and core operations.
- :class:`~scikitplot.annoy.Index` is the **Python facade** that subclasses the backend
  and composes behavior using independent mixins.

-------------------------------------
Mixins only (independence + MRO scan)
-------------------------------------

This diagram focuses only on the mixins so you can quickly confirm there is no
unexpected inheritance between mixins (they should be **independent**).

.. inheritance-diagram::
   scikitplot.annoy._mixins._meta.MetaMixin
   scikitplot.annoy._mixins._io.IndexIOMixin
   scikitplot.annoy._mixins._pickle.PickleMixin
   scikitplot.annoy._mixins._vectors.VectorOpsMixin
   scikitplot.annoy._mixins._ndarray.NDArrayMixin
   scikitplot.annoy._mixins._plotting.PlottingMixin
   :parts: 1

---------------------
Notes and Limitations
---------------------

Inheritance diagrams show **class derivation**, but they do not show **composition**
relationships (e.g., an optional ``self._annoy`` backend attribute). For composition
support, refer to the “glue” helpers documented elsewhere (e.g., ``backend_for(self)``,
``lock_for(self)``).

- An inheritance diagram shows “parent → child” class relationships.
- It helps you understand which class extends another class.

Sphinx can render inheritance diagrams using the built-in extension
``sphinx.ext.inheritance_diagram``. The diagram is rendered using Graphviz
(the ``dot`` tool), so Graphviz must be available in your build environment.

.. note::

   In your ``conf.py``:

   .. code-block:: python

      extensions += ["sphinx.ext.inheritance_diagram"]

See Also
--------
- :mod:`sphinx.ext.inheritance_diagram`
- Graphviz (``dot``) documentation
