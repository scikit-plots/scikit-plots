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

.. currentmodule:: scikitplot.cython

.. _cython-index:

Cython Realtime PKG/MOD Generation
======================================================================

Examples relevant to the :py:mod:`~.cython` module.

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_cython_plot_cython_template.py`: Example usage of
  :func:`~.compile_and_load` using template.

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

  from scikitplot.cython import compile_and_load

  m = compile_and_load("def f(int n):\n    return n*n")
  m.f(10)


.. grid:: 1 1 1 1

   .. grid-item-card::
      :padding: 3
      :shadow: none

      .. python -c 'from scikitplot import cython;cython.generate_sphinx_template_docs("./")'

      **cython templates**
      ^^^
      .. toctree::
        :maxdepth: 2

        _templates/templates_index.rst
