.. currentmodule:: scikitplot.visualkeras

.. _visualkeras-index:

Keras architecture visualization
================================

``scikitplot.visualkeras`` renders Keras/TensorFlow model structure as either a layered architecture diagram or a graph/connectivity diagram. It includes the vendored visualkeras behavior behind the Scikit-Plots namespace.

.. important::

   Visualization is derived from model metadata and framework internals. Test representative models against the exact Keras/TensorFlow version you ship; a rendered diagram is not a validation of model computation or tensor correctness.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started
   layered_view
   graph_view
   styling_and_spacing
   python_api
   dependencies_and_limits
   troubleshooting

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: visualkeras-getting-started
      :link-type: ref

      Render a Keras model with layered_view or graph_view.

   .. grid-item-card:: Layered view
      :link: visualkeras-layered-view
      :link-type: ref

      Dimension scaling, ignored layers, text, spacing, colors, and CNN-style diagrams.

   .. grid-item-card:: Graph view
      :link: visualkeras-graph-view
      :link-type: ref

      Connectivity-oriented visualization for functional/branched models.

   .. grid-item-card:: Styling and spacing
      :link: visualkeras-styling-and-spacing
      :link-type: ref

      Color maps, labels, SpacingDummyLayer, image output, and layout trade-offs.

   .. grid-item-card:: Python API
      :link: visualkeras-python-api
      :link-type: ref

      The three stable top-level exports and dependency behavior.

   .. grid-item-card:: Dependencies and limits
      :link: visualkeras-dependencies-and-limits
      :link-type: ref

      TensorFlow/Keras/Pillow/aggdraw optionality, model-shape caveats, and current maintenance findings.

   .. grid-item-card:: Troubleshooting
      :link: visualkeras-troubleshooting
      :link-type: ref

      Unsupported layer shapes, Keras-version differences, fonts/output files, and large diagrams.

Which page should I use?
------------------------

Start with **Getting started**. Then choose the page that owns the decision
you are making: architecture and lifecycle, configuration, persistence,
performance, public API, or limits. The guide intentionally separates those
concerns so examples do not accidentally become policy.

Examples
--------

See :ref:`visualkeras_examples` for executable examples that accompany this guide.

Further reading
---------------

.. seealso::

   * https://github.com/paulgavrikov/visualkeras
   * https://github.com/triton-lang/triton/
   * https://triton-lang.org/main/getting-started/installation.html
