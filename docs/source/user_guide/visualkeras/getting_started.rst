.. _visualkeras-getting-started:

Getting started
===============

.. code-block:: python

   import scikitplot.visualkeras as visualkeras

   image = visualkeras.layered_view(model)
   image.save("model.png")

Use :func:`layered_view` for a compact architecture-style diagram, especially
for sequential/CNN-like models. Use :func:`graph_view` when connectivity and
branching are more important than layer-block dimensions.

TensorFlow/Keras is intentionally not imported just to import every helper;
rendering a real model still requires a compatible framework environment.
