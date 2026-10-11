.. _annoy-getting-started:

Getting started
===============

Use :class:`Index` for the high-level Scikit-Plots facade. ``AnnoyIndex`` and
``Annoy`` expose the lower-level compiled API.

.. code-block:: python

   from scikitplot.annoy import Index

   index = Index(3, "angular")
   index.add_item(0, [1.0, 0.0, 0.0])
   index.add_item(1, [0.9, 0.1, 0.0])
   index.add_item(2, [0.0, 1.0, 0.0])
   index.build(10)

   ids, distances = index.get_nns_by_item(0, 2, include_distances=True)

The vector length supplied at construction is part of the index contract. The
high-level :class:`Index` also supports lazy dimensionality when constructed
with ``0`` or ``None`` and the first added vector can establish the dimension.
Do not mix vector lengths in one index.

Build and query are different phases
------------------------------------

Annoy is fundamentally build-oriented:

.. code-block:: text

   add_item -> add_item -> ... -> build -> query/query/query

Plan mutations before ``build``. If your application needs continuous online
insertion or deletion, choose a backend designed for mutable indexes rather than
forcing Annoy to behave like one.

Metric choice
-------------

The facade accepts the backend's metric vocabulary and common aliases. Metric
selection changes the geometry of nearest-neighbor search and must be consistent
between index construction and later loading/use. For normalized embedding
vectors, angular/cosine-style search is common; do not select a metric merely
because it produces a numerically smaller distance.
