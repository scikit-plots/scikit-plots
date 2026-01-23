.. docs/source/user_guide/annoy/annoy_index_vector_database.rst
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

.. https://waldyrious.net/rst-playground/
.. https://rst-tutorial.yakimka.me/playground

.. currentmodule:: scikitplot.annoy

.. _annoy_index_vector_database:

======================================================================
Vector Similarity Search and Vector Database
======================================================================

This page explains vector databases in a simple way.

- A **vector database** stores vectors (numbers).
- It can search for *similar vectors* very fast.
- This is useful for AI apps.

A **vector database** stores, manages, and indexes high-dimensional vectors and
is designed for low-latency similarity queries.

Vector databases are popular for AI because they work well with unstructured
data like text, images, and audio (after you convert them into embeddings).


Vector similarity search
------------------------

Vector similarity search finds the items whose vectors are *closest* to a query vector.
It is widely used in retrieval tasks such as semantic search, recommendations, and clustering.

Distance and similarity metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different metrics define "closeness" in different ways:

- **Dot product (inner product)** is often used as a similarity score.

  - Larger :math:`\mathbf{u}\cdot\mathbf{v}` means more similar.
  - Some libraries expect a *distance* to minimize, so they use the **negative dot product**.

- **Cosine similarity** measures the *direction* (angle) between vectors.

  - Range: **-1 to 1** (for many embedding use cases, values are often **0 to 1**).
  - **1** means vectors point in the same direction (most similar).
  - **0** means vectors are orthogonal (no directional similarity).
  - **-1** means vectors point in opposite directions (most dissimilar).

- **Cosine distance** converts cosine similarity into a distance.

  - A common definition is :math:`1 - \text{cosine\_similarity}`.
  - Range (with this definition): **0 to 2**.
  - **0** means most similar; values closer to **2** mean more dissimilar.

- **Euclidean (L2) distance** measures straight-line distance.

  - Think: *as-the-crow-flies* distance in space.
  - Larger values mean vectors are farther apart.

- **Manhattan (L1) distance** measures grid-like distance.

  - Think: moving along city blocks (right/left/up/down).
  - Often more robust to outliers than L2.

- **Hamming distance** counts how many positions differ.

  - Used for **binary vectors** (0/1) or **equal-length strings**.
  - It is the number of indices where :math:`u_i \neq v_i`.

Formulas
~~~~~~~~

For two vectors :math:`\mathbf{u}` and :math:`\mathbf{v}` of length :math:`k`:

.. note::

   Dot product is *not* scale-invariant: if you multiply :math:`\mathbf{u}` by 2,
   the dot product doubles. Using :math:`d_{\text{dot}} = -(\mathbf{u}\cdot\mathbf{v})`
   produces the same ranking as maximizing :math:`\mathbf{u}\cdot\mathbf{v}` (it just
   flips the sign). If vectors are L2-normalized, dot product and cosine similarity
   become equivalent.

- **Dot product (similarity score)**:

  .. math::

     s_{\text{dot}}(\mathbf{u}, \mathbf{v})
     = \mathbf{u} \cdot \mathbf{v}
     = \sum_{i=1}^{k} u_i v_i

- **Negative dot product** (dot as a distance to minimize):

  .. math::

     d_{\text{dot}}(\mathbf{u}, \mathbf{v})
     = -(\mathbf{u} \cdot \mathbf{v})
     = -\sum_{i=1}^{k} u_i v_i

- **Cosine similarity**:

  .. math::

     \text{cos\_sim}(\mathbf{u}, \mathbf{v})
     = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert}

- **Cosine distance** (common definition):

  .. math::

     d_{\text{cos}}(\mathbf{u}, \mathbf{v})
     = 1 - \text{cos\_sim}(\mathbf{u}, \mathbf{v})

- **Euclidean (L2) distance**:

  .. math::

     d_{2}(\mathbf{u}, \mathbf{v})
     = \sqrt{\sum_{i=1}^{k} (u_i - v_i)^2}

- **Manhattan (L1) distance**:

  .. math::

     d_{1}(\mathbf{u}, \mathbf{v})
     = \sum_{i=1}^{k} \lvert u_i - v_i \rvert

- **Hamming distance**:

  .. math::

     d_{\text{ham}}(\mathbf{u}, \mathbf{v})
     = \sum_{i=1}^{k} \mathbb{1}[u_i \neq v_i]


Vector database vs “vector index library”
-----------------------------------------

A *vector index library* (example: Annoy) is usually a library that you run
inside your application process.

A *vector database* is usually a separate service (or a database extension)
that focuses on:

- storing vectors + metadata
- indexing vectors for fast search
- scaling to large datasets and many users
- operational features (replication, backups, monitoring, access control)

Vector databases store vectors (example: pgvector with PostgreSQL) and support similarity search,
often using approximate nearest neighbor methods in a pipeline for fast retrieval.

Pros and cons of vector search
------------------------------

Vector similarity search can be very effective:

- Efficient searching with special index structures (fast retrieval)
- High accuracy for semantic similarity (meaning-based matches)
- Range queries (search within a threshold)

But there are also limitations:

- High-dimensional data can be hard (needs special handling)
- Scalability can be challenging for very large datasets
- Distance metric choice matters (wrong metric = bad results)
- Indexing/storage needs can be high (large vectors take space)

5 practical tips
----------------

Instaclustr suggests these practical steps for good results:

1. Clean and normalize data (reduce noise; keep a common scale)
2. Configure and tune algorithms (balance speed and accuracy)
3. Use sharding / partitioning for large datasets
4. Consider hardware acceleration (GPU/TPU) when needed
5. Handle high-dimensional data (e.g., dimensionality reduction when useful)

Open source options
-------------------

Instaclustr lists popular open source options including:

Dedicated / vector-native options (examples)

- Elasticsearch
- Faiss
- Qdrant
- OpenSearch
- Chroma
- Milvus
- Weaviate

General-purpose databases with vector support (examples)

- PostgreSQL (via extensions such as pgvector)
- Others depending on your stack


How to choose (simple rules)
----------------------------

Choose a vector index library (like Annoy) when:

- you want something small and local
- you control the process memory
- you can rebuild the index when needed

Choose a vector database when:

- you need a shared service for many users/apps
- you need storage + metadata filters + operations (backup/monitoring)
- you need easy scaling and high availability
