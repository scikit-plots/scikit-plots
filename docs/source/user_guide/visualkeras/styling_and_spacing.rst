.. _visualkeras-styling-and-spacing:

Styling and spacing
===================

Color maps and text callbacks can make architecture roles visible without
changing the underlying model. Keep styling deterministic in generated docs by
using explicit colors/fonts/layout parameters where visual diffs matter.

:class:`SpacingDummyLayer` creates a dynamic Keras-compatible placeholder used
only to introduce visual separation. It imports TensorFlow when instantiated.
Do not add spacing layers to a production computational graph merely to obtain a
nicer documentation image; build a visualization-specific representation when
possible.
