.. _visualkeras-dependencies-and-limits:

Dependencies and limits
=======================

Rendering can involve Keras/TensorFlow plus Pillow and optional drawing/font
backends. Missing optional packages should be diagnosed at the operation that
needs them rather than turning package import into an eager dependency check.

The maintenance snapshot currently records open findings around graph-shape
handling, ellipsizing, optional backends, default/deprecation behavior and test
matrix coverage. Treat representative gallery output as examples, not proof
that every custom layer/framework version is supported.

Large models can create very large images or unreadable graphs. Use filtering,
scaling and focused sub-model diagrams when a single “whole network” picture
stops communicating useful structure.
