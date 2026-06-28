:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-embedding-similarity:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">🧬&nbsp;&nbsp;<b>Embedding Similarity</b></div>`

======================
Embedding Similarity
======================

*Measuring how close two items are in a learned embedding space.*

What it is
----------

**Embedding similarity** is how we measure the closeness of two objects once they
have been turned into embedding vectors: similar objects should have vectors that
are close, so a similarity (or distance) score on the vectors becomes a proxy for
how alike the *objects* are. It is the operation that makes embeddings useful in
practice — search, recommendation and clustering are all "find the nearby
vectors" at heart.

Similarity measures
-------------------

cosine similarity
^^^^^^^^^^^^^^^^^^

The most common choice for semantic vectors. It measures the angle between two
vectors and ignores their magnitude, so it compares *direction* (meaning) rather
than length:

.. math::

   \operatorname{cosine}(u, v) = \frac{u \cdot v}{\lVert u\rVert\,\lVert v\rVert}
   \quad\in\; [-1, 1],

where 1 means identical direction, 0 means unrelated and −1 means opposite.

other measures
^^^^^^^^^^^^^^

- **Dot product** — like cosine but sensitive to magnitude; common inside neural
  networks (attention, matrix factorisation).
- **Euclidean distance (L2)** — straight-line distance; smaller is more similar.
- **Manhattan distance (L1)** — sum of absolute coordinate differences.
- **Jaccard similarity** — for sparse sets rather than dense vectors.
- **Mahalanobis distance** — accounts for feature covariance.

Why it matters
--------------

Reducing "are these two things alike?" to a number on vectors is what powers
semantic search, "people who liked this also liked that" recommendations, face and
image retrieval, and clustering or anomaly detection in embedding space.

In practice
-----------

.. code-block:: python

   import numpy as np
   from sklearn.metrics.pairwise import cosine_similarity

   u = np.array([[0.1, 0.8, 0.5]])
   v = np.array([[0.2, 0.7, 0.4]])

   score = cosine_similarity(u, v)[0, 0]
   print(f"cosine similarity: {score:.2f}")   # ~0.99  ->  very similar

----

**Mind map — connected ideas**

   :doc:`Embedding <173-embedding>` · :doc:`Autoencoder <171-autoencoder>`

----

**More in Representations & Embeddings**

   :doc:`Autoencoder <171-autoencoder>` · :doc:`Embedding <173-embedding>` · :doc:`Frozen Encoder <172-frozen-encoder>`

----

*Theme:* :ref:`Representations & Embeddings <term-theme-repr>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Embedding Similarity <https://insightful-data-lab.com/2025/08/20/embedding-similarity/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
