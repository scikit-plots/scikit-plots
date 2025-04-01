.. _visualkeras-index:

======================================================================
Visualkeras (experimental)
======================================================================

This module contains functions related to :py:mod:`~.visualkeras`.
For visualkeras for Keras / TensorFlow Models :py:class:`~tensorflow.keras.Model`.

.. seealso::

   * https://github.com/paulgavrikov/visualkeras

Documentation is available in the docstrings and
online at https://github.com/paulgavrikov/visualkeras.

.. .. code-block:: python

.. prompt:: bash >>> auto

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
