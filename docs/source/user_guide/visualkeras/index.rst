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

.. currentmodule:: scikitplot.visualkeras

.. _visualkeras-index:

======================================================================
Visualkeras
======================================================================

This module contains functions related to :py:mod:`~.visualkeras`.
For visualkeras for Keras / TensorFlow Models :py:class:`~tensorflow.keras.Model`.

.. seealso::

   * https://github.com/paulgavrikov/visualkeras

Documentation is available in the docstrings and
online at https://github.com/paulgavrikov/visualkeras.

.. .. jupyter-execute
.. .. code-block:: python
.. prompt:: python >>>

   # (optionally)
   import visualkeras

   # (recommended) scikit-plots also include visualkeras
   import scikitplot.visualkeras as visualkeras
   from scikitplot import visualkeras


Visualkeras Visualization
----------------------------------------------------------------------

Visualkeras is a Python package to help visualize Keras (either standalone
or included in tensorflow) neural network architectures.
It allows easy styling to fit most needs. This module supports layered style
architecture generation which is great for CNNs (Convolutional Neural Networks),
and a graph style architecture, which works great for most models
including plain feed-forward networks.


Graphical Visualization
----------------------------------------------------------------------

This module contains functions related to :py:func:`~.graph_view`.


Layered Visualization
----------------------------------------------------------------------

This module contains functions related to :py:func:`~.layered_view`.


Visualization Helper
----------------------------------------------------------------------

This module contains functions related to :py:class:`~.SpacingDummyLayer`.
