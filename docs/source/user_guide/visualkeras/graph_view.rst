.. _visualkeras-graph-view:

Graph view
==========

:func:`graph_view` builds a connectivity-oriented representation from the Keras
model graph. It is a better fit for functional models with branches, merges or
multiple pathways.

Graph extraction depends on Keras/TensorFlow layer/node metadata whose internal
representation has changed across framework releases. Treat framework-version
compatibility as part of your visualization test matrix rather than assuming a
single graph traversal path works forever.
